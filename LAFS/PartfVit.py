import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Mobilenet import MobileNetV3_backbone
from .utils import Transformer, extract_patches_pytorch_gridsample
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

# Part-fViT Model (Landmark-Aware ViT)

MIN_NUM_PATCHES = 15

class ViT_face_landmark_patch8(nn.Module):
    """
    Part-fViT: Landmark-based Facial Vision Transformer
    
    Uses MobileNetV3 to predict facial landmarks, then extracts patches
    at those landmark locations for the transformer to process.
    """
    def __init__(
        self,
        image_size=112,
        patch_size=8,
        dim=768,
        depth=12,
        heads=11,
        mlp_dim=2048,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        num_patches=196  # 14x14 grid
    ):
        super().__init__()
        
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'Number of patches ({num_patches}) too small'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.row_num = int(np.sqrt(num_patches))  # 14 for 196 patches
        self.dim = dim
        
        # Landmark detection network (MobileNetV3)
        self.stn = MobileNetV3_backbone(mode='large')
        
        # Output layer: MobileNetV3 features (160-dim) -> landmark coordinates (14*14*2=392)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(160, self.row_num * self.row_num * 2),
        )
        
        # Patch shape for extraction
        self.patch_shape = torch.tensor([patch_size, patch_size])
        self.theta = 0
        
        # Transformer components
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()
        self.sigmoid = nn.Sigmoid()
        
        # Final normalization
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.cls_token, std=.02)
    
    def forward(self, x, visualize=False):
        p = self.patch_size
        
        # 1. Get landmark features from MobileNetV3
        theta0 = self.stn(x)
        theta0 = theta0.mean(dim=(-2, -1))  # Global average pooling
        
        # 2. Predict landmark coordinates
        theta = self.output_layer(theta0)
        
        # 4. Normalize landmarks to image coordinates (0-111)
        t_max = torch.max(theta, 1)[0]
        t_max = torch.unsqueeze(t_max, dim=1).repeat(1, self.row_num * self.row_num * 2)
        t_min = torch.min(theta, 1)[0]
        t_min = torch.unsqueeze(t_min, dim=1).repeat(1, self.row_num * self.row_num * 2)
        theta = (theta - t_min) / (t_max - t_min + 1e-8) * 111
        
        # Reshape to [batch, num_landmarks, 2]
        theta = theta.view(-1, self.row_num * self.row_num, 2)
        self.theta = theta
        theta_detached = theta.detach()
        
        num_land = self.row_num * self.row_num
        
        # 5. Extract patches at landmark locations
        x = extract_patches_pytorch_gridsample(
            x, theta_detached[:, :num_land],
            patch_shape=self.patch_shape.to(x.device),
            num_landm=num_land
        )
        
        # 6. Convert patches to embeddings
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        
        # 7. Add cls token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 8. Add positional embeddings
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # 9. Transformer forward pass
        x = self.transformer(x)
        
        # 10. Pool and normalize
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        
        if visualize:
            return emb, self.theta
        return emb