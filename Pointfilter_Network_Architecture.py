import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    """Geometry-aware attention with relative position encoding"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projection layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative position encoding
        self.rel_pos_embedding = nn.Parameter(torch.randn(num_heads, 64))
        
    def get_relative_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute relative distances between points"""
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B,N,N,3)
        return torch.norm(rel_pos, dim=-1)  # (B,N,N)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
               positions: torch.Tensor) -> torch.Tensor:
        B, N, _ = query.shape
        
        # Project queries/keys/values
        Q = self.w_q(query).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias
        rel_dist = self.get_relative_positions(positions)  # (B,N,N)
        rel_dist_idx = torch.clamp((rel_dist / 0.1).long(), 0, 63)  # Discretize
        pos_bias = self.rel_pos_embedding[:, rel_dist_idx]  # (H,B,N,N)
        scores = scores + pos_bias.permute(1, 0, 2, 3)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Combine heads
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.w_o(context)

class EdgeConvolutionLayer(nn.Module):
    """Local feature extraction with dynamic graph CNN"""
    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def knn(self, x: torch.Tensor) -> torch.Tensor:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_dist = -xx - inner - xx.transpose(2, 1)
        return pairwise_dist.topk(k=self.k, dim=-1)[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        idx = self.knn(x)
        B, C, N = x.shape
        
        # Gather neighbors
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1)*N
        idx = (idx + idx_base).view(-1)
        x = x.transpose(2, 1).contiguous()
        neighbors = x.view(B*N, -1)[idx, :].view(B, N, self.k, C)
        
        # Edge features
        x = x.view(B, N, 1, C).repeat(1, 1, self.k, 1)
        edge_feat = torch.cat([neighbors - x, x], dim=3).permute(0, 3, 1, 2)
        
        return self.conv(edge_feat).max(dim=-1, keepdim=False)[0]

class GeometryAwareLayer(nn.Module):
    """Single layer of geometry-aware transformer"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x, x, x, pos)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class GeometryAwareTransformer(nn.Module):
    """Stack of geometry-aware layers"""
    def __init__(self, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometryAwareLayer(d_model, num_heads) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos)
        return x

class TDNetDenoiser(nn.Module):
    """Complete denoising network with geometry-aware transformer"""
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        # Local feature extraction
        self.edge_conv1 = EdgeConvolutionLayer(3, 64)
        self.edge_conv2 = EdgeConvolutionLayer(64, 128)
        self.input_proj = nn.Linear(128, d_model)
        
        # Geometry-aware processing
        self.transformer = GeometryAwareTransformer(d_model, num_heads, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 3)  # 3D displacement
        )

    def forward(self, noisy_points: torch.Tensor) -> torch.Tensor:

        assert noisy_points.shape[-1] == 3, "Input must be [B,N,3]"
        assert noisy_points.shape[1] == 500, "Input must have 500 points"

        
        # 1. Extract local features
        x = noisy_points.transpose(1, 2)
        x = self.edge_conv1(x)
        x = self.edge_conv2(x)
        x = x.transpose(1, 2)
        features = self.input_proj(x)
        
        # 2. Global geometry processing
        features = self.transformer(features, noisy_points)
        
        # 3. Predict displacement
        displacement = self.output_proj(features)
        return noisy_points + displacement

class ChamferLoss(nn.Module):
    """Simplified loss using only Chamfer distance"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(pred, target)  # (B,N,M)
        return torch.min(dist, dim=2)[0].mean() + torch.min(dist, dim=1)[0].mean()