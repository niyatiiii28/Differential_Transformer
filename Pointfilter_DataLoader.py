from __future__ import print_function
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import os
import numpy as np
import scipy.spatial as sp
from Pointfilter_Utils import pca_alignment, add_noise_to_batch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Return empty tensors with proper shape
        return {
            'noisy_points': torch.empty((0, 500, 3)),  # matches points_per_patch
            'gt_points': torch.empty((0, 500, 3))
        }
    return {
        'noisy_points': torch.stack([item['noisy_points'] for item in batch]),
        'gt_points': torch.stack([item['gt_points'] for item in batch])
    }

class PointcloudPatchDataset(data.Dataset):
    def __init__(self, root=None, shapes_list_file=None, patch_radius=0.05, 
                 points_per_patch=500, seed=None, train_state='train', 
                 shape_name=None, noise_type='gaussian', noise_level=0.02,
                 use_augmentation=True, scale_range=[0.9, 1.1]):
        
        self.root = root
        self.shapes_list_file = shapes_list_file
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.seed = seed
        self.train_state = train_state
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.use_augmentation = use_augmentation
        self.scale_range = scale_range

        if self.seed is None:
            self.seed = np.random.randint(0, 2**10 - 1)
        self.rng = np.random.RandomState(self.seed)

        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.gt_shapes = []
        self.noise_shapes = []
        self.shape_names = []

        if self.train_state == 'evaluation' and shape_name is not None:
            self._load_evaluation_data(shape_name)
        elif self.train_state == 'train':
            self._load_training_data()
   

    def shape_index(self, index):
        """Convert flat index into (shape_index, patch_index) tuple"""
        if not hasattr(self, 'shape_patch_count'):
            raise RuntimeError("Dataset not properly initialized - run _load_training_data or _load_evaluation_data first")
        
        offset = 0
        for shape_idx, patch_count in enumerate(self.shape_patch_count):
            if index < offset + patch_count:
                return shape_idx, index - offset
            offset += patch_count
        raise IndexError(
            f"Index {index} is out of bounds. "
            f"Dataset contains {offset} patches across {len(self.shape_patch_count)} shapes."
        )

    def _load_evaluation_data(self, shape_name):
        """Load single shape for evaluation"""
        pts_path = os.path.join(self.root, shape_name + '.npy')
        if not os.path.exists(pts_path):
            raise FileNotFoundError(f"Point file not found {pts_path}")

        pts = np.load(pts_path)
        kdtree = sp.cKDTree(pts)
        self.noise_shapes.append({'pts': pts, 'kdtree': kdtree})
        self.shape_patch_count.append(pts.shape[0])
        bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2))
        self.patch_radius_absolute.append(bbdiag * self.patch_radius)
        self.shape_names.append(shape_name)

    def _load_training_data(self):
        """Load training dataset with synthetic noise"""
        with open(os.path.join(self.root, self.shapes_list_file)) as f:
            self.shape_names = [x.strip() for x in f.readlines() if x.strip()]
        
        for shape_name in self.shape_names:
            pts_path = os.path.join(self.root, shape_name + '.npy')
            if not os.path.exists(pts_path):
                continue
                
            pts = np.load(pts_path)
            if pts.shape[0] == 0:
                continue
                
            # Create clean+kdtree and noisy versions
            kdtree = sp.cKDTree(pts)
            self.gt_shapes.append({'pts': pts, 'kdtree': kdtree})
            
            noisy_pts = add_noise_to_batch(
                torch.from_numpy(pts).float(),
                noise_type=self.noise_type,
                noise_level=self.noise_level
            ).numpy()
            
            self.noise_shapes.append({
                'pts': noisy_pts,
                'kdtree': sp.cKDTree(noisy_pts)
            })
            
            self.shape_patch_count.append(pts.shape[0])
            bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag * self.patch_radius)

    def _apply_augmentation(self, patch_pts):
        """Apply random scaling/rotation"""
        if not self.use_augmentation:
            return patch_pts
            
        # Random scaling
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1], size=3)
        patch_pts = patch_pts * scale
        
        # Random z-rotation
        angle = self.rng.uniform(0, 2*np.pi)
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return patch_pts @ rot.T
        
    def __getitem__(self, index):
        shape_ind, patch_ind = self.shape_index(index)
        patch_radius = self.patch_radius_absolute[shape_ind]
        center = self.noise_shapes[shape_ind]['pts'][patch_ind]
        
        # 1. Get noisy patch points (RANDOM SAMPLING)
        noise_idx = self.noise_shapes[shape_ind]['kdtree'].query_ball_point(center, patch_radius)
        if len(noise_idx) < 3:  # Need at least 3 points for PCA
            return None
        
        # Get ALL noisy points first
        noise_patch = self.noise_shapes[shape_ind]['pts'][noise_idx] - center
        noise_patch = self._apply_augmentation(noise_patch)
        
        try:
            noise_patch, inv_transform = pca_alignment(noise_patch)
        except np.linalg.LinAlgError:
            return None
        
        noise_patch /= patch_radius  # Normalize
        
        # RANDOMLY SAMPLE exactly 500 points (with duplication if needed)
        available_points = noise_patch.shape[0]
        if available_points < self.points_per_patch:
            # Duplicate points if we don't have enough
            repeat_times = (self.points_per_patch // available_points) + 1
            sample_idx = np.tile(np.arange(available_points), repeat_times)[:self.points_per_patch]
        else:
            # Randomly sample without replacement if we have enough points
            sample_idx = self.rng.choice(available_points, size=self.points_per_patch, replace=False)
        
        noise_patch = noise_patch[sample_idx]  # This is now guaranteed to work

        # 2. Get corresponding GT patch (must use same spatial locations)
        gt_idx = self.gt_shapes[shape_ind]['kdtree'].query_ball_point(center, patch_radius)
        if len(gt_idx) < 3:
            return None
        
        gt_patch = self.gt_shapes[shape_ind]['pts'][gt_idx] - center
        gt_patch = np.array(inv_transform @ gt_patch.T).T  # Apply same transform
        gt_patch /= patch_radius  # Same normalization
        
        # Find nearest GT points for our sampled noise points
        gt_kdtree = sp.cKDTree(gt_patch)
        _, gt_sample_idx = gt_kdtree.query(noise_patch, k=1)  # Find closest GT point for each noise point
        
        # Final check to prevent any indexing errors
        gt_sample_idx = np.clip(gt_sample_idx, 0, len(gt_patch) - 1)
        gt_patch = gt_patch[gt_sample_idx]
        
        return {
            'noisy_points': torch.FloatTensor(noise_patch),
            'gt_points': torch.FloatTensor(gt_patch)
        }
        

class RandomPointcloudPatchSampler:
    """Sampler that randomly selects patches with uniform probability"""
    def __init__(self, data_source, patches_per_shape, seed=None):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed or np.random.randint(0, 2**32-1)
        self.rng = np.random.RandomState(self.seed)
        
    def __iter__(self):
        population = sum(self.data_source.shape_patch_count)
        if population == 0:
            return iter([])
        return iter(self.rng.choice(population, size=min(self.patches_per_shape, population), replace=False))
        
    def __len__(self):
        return min(self.patches_per_shape, sum(self.data_source.shape_patch_count))