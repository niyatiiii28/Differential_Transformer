import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import open3d as o3d
import argparse
from typing import Optional
from Pointfilter_Network_Architecture import ChamferLoss

########################## Parameters ########################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument('--name', type=str, default='pcdenoising', help='training run name')
    parser.add_argument('--network_model_dir', type=str, default='./Summary/Models/Train', help='output folder (trained models)')
    parser.add_argument('--trainset', type=str, default='./Dataset/Train', help='training set file name')
    parser.add_argument('--testset', type=str, default='./Dataset/Test', help='testing set file name')
    parser.add_argument('--save_dir', type=str, default='./Dataset/Results', help='results directory')
    parser.add_argument('--summary_dir', type=str, default='./Summary/Models/Train/logs', help='tensorboard logs directory')

    # training parameters
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--manualSeed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--patch_per_shape', type=int, default=8000, help='patches per shape')
    parser.add_argument('--patch_radius', type=float, default=0.05, help='patch radius relative to shape diameter')

    # optimization parameters
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--model_interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')
    
    # noise parameters (optional for data augmentation)
    parser.add_argument('--noise_type', type=str, default='none', 
                       choices=['none', 'gaussian', 'uniform', 'sparse', 'mixed'], 
                       help='type of noise to add')
    parser.add_argument('--noise_level', type=float, default=0.02, 
                       help='magnitude of noise to add')
    parser.add_argument('--corruption_rate', type=float, default=0.1,
                       help='fraction of points to corrupt for sparse noise')
    
    # augmentation parameters
    parser.add_argument('--use_augmentation', type=str2bool, default=True, help='enable data augmentation')
    parser.add_argument('--scale_range', type=float, nargs=2, default=[0.9, 1.1], help='range for random scaling')

    return parser.parse_args()

################### Pre-Processing Tools ########################

def get_principle_dirs(pts):
    """Compute principal directions using PCA"""
    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)
    return principle_dirs

def pca_alignment(pts, random_flag=False):
    """Align point cloud using PCA"""
    pca_dirs = get_principle_dirs(pts)

    if random_flag:
        pca_dirs *= np.random.choice([-1, 1], 1)

    rotate_1 = compute_rotation_matrix(pca_dirs[2], [0, 0, 1], pca_dirs[1])
    pca_dirs = np.array(rotate_1 * pca_dirs.T).T
    rotate_2 = compute_rotation_matrix(pca_dirs[1], [1, 0, 0], pca_dirs[2])
    pts = np.array(rotate_2 * rotate_1 * np.matrix(pts.T)).T

    inv_rotation = np.array(np.linalg.inv(rotate_2 * rotate_1))
    return pts, inv_rotation

def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_rotation_matrix(src_vec, dest_vec, src_vertical_vec=None):
    """Compute rotation matrix between two vectors"""
    if np.linalg.norm(np.cross(src_vec, dest_vec), 2) == 0 or np.abs(np.dot(src_vec, dest_vec)) >= 1.0:
        if np.dot(src_vec, dest_vec) < 0:
            return rotation_matrix(src_vertical_vec, np.pi)
        return np.identity(3)
    
    alpha = np.arccos(np.dot(src_vec, dest_vec))
    axis = np.cross(src_vec, dest_vec) / np.linalg.norm(np.cross(src_vec, dest_vec), 2)
    
    # Rodrigues' rotation formula
    cos = np.cos(alpha)
    sin = np.sin(alpha)
    t = 1 - cos
    
    R = np.array([
        [t*axis[0]*axis[0] + cos, t*axis[0]*axis[1] - sin*axis[2], t*axis[0]*axis[2] + sin*axis[1]],
        [t*axis[1]*axis[0] + sin*axis[2], t*axis[1]*axis[1] + cos, t*axis[1]*axis[2] - sin*axis[0]],
        [t*axis[2]*axis[0] - sin*axis[1], t*axis[2]*axis[1] + sin*axis[0], t*axis[2]*axis[2] + cos]
    ])
    
    return np.matrix(R)

def rotation_matrix(axis, theta):
    """Generate rotation matrix from axis-angle representation"""
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    
    return np.matrix(np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ]))

################### Loss Functions ########################



class RobustChamferLoss(nn.Module):
    """L1 Chamfer Distance for outlier robustness"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        dist = torch.cdist(pred, target, p=1)  # L1 distance
        return torch.min(dist, dim=2)[0].mean() + torch.min(dist, dim=1)[0].mean()

################### Noise Generation (Optional) ########################

def add_gaussian_noise(points, noise_level=0.02):
    noise = torch.randn_like(points) * noise_level
    return points + noise

def add_noise_to_batch(batch, noise_type='none', noise_level=0.02, corruption_rate=0.1):
    if noise_type == 'none':
        return batch
    elif noise_type == 'gaussian':
        return add_gaussian_noise(batch, noise_level)
    # (Other noise types remain unchanged)

################### Data Augmentation Tools ########################

def random_rotate_pointcloud(points):
    """Randomly rotate the point cloud"""
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cosval, sinval, 0],
        [-sinval, cosval, 0],
        [0, 0, 1]
    ])
    return torch.from_numpy(np.dot(points.numpy(), rotation_matrix.T))

def random_scale_pointcloud(points, scale_range=(0.9, 1.1)):
    scale = torch.rand(3, device=points.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    return points * scale.unsqueeze(0)

def augment_pointcloud_batch(batch, use_scale=True, use_rotation=True):
    if use_scale:
        batch = random_scale_pointcloud(batch)
    if use_rotation:
        batch = random_rotate_pointcloud(batch)
    return batch

################### Training Utilities ########################

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint
    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

################### Evaluation Metrics ########################

def compute_metrics(pred, target):
    return {
        'chamfer': ChamferLoss()(pred, target),
        'chamfer_l1': RobustChamferLoss()(pred, target)
    }

################### Patch Sampling ########################

def farthest_point_sampling(points, num_samples):
    device = points.device
    B, N, _ = points.shape
    
    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distances = torch.ones(B, N, device=device) * 1e10
    
    farthest = torch.randint(0, N, (B,), device=device)
    
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, dim=-1)[1]
    
    return centroids