# coding=utf-8
from __future__ import print_function
from tensorboardX import SummaryWriter
from Pointfilter_Network_Architecture import TDNetDenoiser, ChamferLoss
from Pointfilter_DataLoader import PointcloudPatchDataset, RandomPointcloudPatchSampler, collate_fn
from Pointfilter_Utils import parse_arguments, adjust_learning_rate
import torch.nn as nn
import os
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import open3d as o3d 
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def save_loss_graph(loss_history, filename="loss_vs_epoch.png"):
    """Save training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['train'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Chamfer Distance')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Loss graph saved to {filename}")

def train(opt):
    # Setup directories
    os.makedirs(opt.summary_dir, exist_ok=True)
    os.makedirs(opt.network_model_dir, exist_ok=True)
    
    # Set seeds
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    # Initialize model and loss
    model = TDNetDenoiser().cuda()
    criterion = ChamferLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    
    # Tensorboard
    writer = SummaryWriter(opt.summary_dir)
    
    # Load checkpoint if resuming
    if hasattr(opt, 'resume') and opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> Loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{opt.resume}'")
    else:
        print("=> No checkpoint to resume - training from scratch")

    # Dataset and dataloader
    train_dataset = PointcloudPatchDataset(
        root=opt.trainset,
        shapes_list_file='train.txt',
        patch_radius=0.05,
        points_per_patch=500,
        noise_type=opt.noise_type,
        noise_level=opt.noise_level,
        use_augmentation=opt.use_augmentation,
        scale_range=opt.scale_range
    )
    
    train_sampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=opt.patch_per_shape
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.batchSize,
        collate_fn=collate_fn,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True
    )

    # Training loop
    best_loss = float('inf')
    loss_history = {'train': []}
    
    for epoch in range(opt.start_epoch, opt.nepoch):
        model.train()
        epoch_loss = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{opt.nepoch}')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to GPU
            noisy = batch_data['noisy_points'].cuda(non_blocking=True)
            gt = batch_data['gt_points'].cuda(non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            denoised = model(noisy)
            
            # Compute loss
            loss = criterion(denoised, gt)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update stats
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
            # Tensorboard logging
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), global_step)
        
        # Epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        loss_history['train'].append(avg_loss)
        print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}')
        
        # Save checkpoint
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss
        }
        
        # Save regular checkpoint
        torch.save(state, os.path.join(opt.network_model_dir, 'checkpoint.pth'))
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, os.path.join(opt.network_model_dir, 'best_model.pth'))
        
        # Save periodic snapshot
        if (epoch + 1) % opt.model_interval == 0:
            torch.save(state, os.path.join(opt.network_model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save loss curve
        if (epoch + 1) % 5 == 0:
            save_loss_graph(loss_history, os.path.join(opt.summary_dir, f'loss_epoch_{epoch+1}.png'))
    
    # Save final model
    torch.save(state, os.path.join(opt.network_model_dir, 'final_model.pth'))
    writer.close()

def prepare_dataset(opt):
    """Automatically create train.txt and handle both Kaggle and local paths"""
    # Determine the base data path (Kaggle or local)
    base_path = opt.trainset if opt.trainset else "./Dataset/Train"
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Path for train.txt
    train_txt_path = os.path.join(base_path, "train.txt")
    
    # Create train.txt if missing or empty
    if not os.path.exists(train_txt_path) or os.path.getsize(train_txt_path) == 0:
        print(f"Creating/updating train.txt at {train_txt_path}")
        npy_files = [f for f in os.listdir(base_path) 
                    if f.endswith('.npy') and not f.endswith('_normal.npy')]
        
        if not npy_files:
            available_files = "\n".join(os.listdir(base_path))
            raise FileNotFoundError(
                f"No .npy files found in {base_path}\n"
                f"Available files:\n{available_files}"
            )
        
        with open(train_txt_path, 'w') as f:
            for npy_file in sorted(npy_files):  # Sort for consistency
                f.write(npy_file.replace('.npy', '') + '\n')
        print(f"Created train.txt with {len(npy_files)} shapes")

    # Normal estimation (if Open3D available)
    shape_names = []
    with open(train_txt_path) as f:
        shape_names = [x.strip() for x in f.readlines() if x.strip()]

    for name in shape_names:
        normal_path = os.path.join(base_path, f"{name}_normal.npy")
        if not os.path.exists(normal_path):
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.load(os.path.join(base_path, f"{name}.npy")))
                pcd.estimate_normals()
                np.save(normal_path, np.asarray(pcd.normals))
            except ImportError:
                print("Open3D not available - skipping normal estimation")
            except Exception as e:
                print(f"Error estimating normals for {name}: {str(e)}")

if __name__ == '__main__':
    opt = parse_arguments()
    
    # Default paths
    opt.trainset = opt.trainset or './Dataset/Train'
    opt.summary_dir = opt.summary_dir or './Summary/Train/logs'
    opt.network_model_dir = opt.network_model_dir or './Summary/Train'

    opt.resume = './Summary/Models/Train/best_model.pth'

    # Prepare dataset (convert formats)
    prepare_dataset(opt)
    
    # Start training
    train(opt)