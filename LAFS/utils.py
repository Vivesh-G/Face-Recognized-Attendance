import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

# Transformer Building Blocks

class Residual_droppath(nn.Module):
    def __init__(self, fn, drop_path_rate=0.1):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    
    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(x, **kwargs)) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., last_dim=None):
        super().__init__()
        if last_dim is None:
            last_dim = dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, last_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.attention_score = 0

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        self.attention_score = attn.detach()
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_droppath(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual_droppath(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

# Landmark-Based Patch Extraction

def extract_patches_pytorch_gridsample(imgs, landmarks, patch_shape, num_landm=196):
    """
    Extracts patches from images at landmark locations using grid sampling.
    
    Args:
        imgs: tensor of shape [batch_size, channels, height, width]
        landmarks: tensor of shape [batch_size, num_landmarks, 2] with (x, y) coordinates
        patch_shape: tensor of [patch_width, patch_height]
        num_landm: number of landmarks/patches to extract
    
    Returns:
        tensor of extracted patches arranged in a grid
    """
    device = landmarks.device
    img_shape = imgs.shape[2]
    
    list_patches = []
    patch_half_shape = patch_shape / 2
    start = -patch_half_shape
    end = patch_half_shape
    
    # Create sampling grid
    sampling_grid = torch.meshgrid(
        torch.arange(start[0], end[0]),
        torch.arange(start[1], end[1]),
        indexing='ij'
    )
    sampling_grid = torch.stack(sampling_grid, dim=0).to(device)
    sampling_grid = torch.transpose(torch.transpose(sampling_grid, 0, 2), 0, 1)
    
    # Extract patch at each landmark
    for i in range(num_landm):
        land = landmarks[:, i, :]
        patch_grid = (sampling_grid[None, :, :, :] + land[:, None, None, :]) / (img_shape * 0.5) - 1
        sing_land_patch = F.grid_sample(imgs, patch_grid, align_corners=False)
        list_patches.append(sing_land_patch)
    
    # Stack and reshape patches into grid
    list_patches = torch.stack(list_patches, dim=2)
    B, c, patches_num, w, h = list_patches.shape
    row = int(np.sqrt(patches_num))
    list_patches = list_patches.reshape(B, c, row, row, w, h)
    list_patches = list_patches.permute(0, 1, 2, 4, 3, 5)
    list_patches = list_patches.reshape(B, c, w * row, h * row)
    
    return list_patches