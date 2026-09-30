import math
import os

import cv2
import numpy as np
import torch
import torch._tensor
import torch.nn as nn
import torch.nn.functional as F

if not hasattr(torch._tensor, '_rebuild_from_type_v2'):
    def _rebuild_from_type_v2(func, new_type, args, state):
        ret = func(*args)
        if type(ret) is not new_type:
            ret = ret.as_subclass(new_type)
        if state is not None:
            ret.__dict__.update(state)
        return ret
    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2

_original_interpolate = F.interpolate
def _patched_interpolate(*args, **kwargs):
    if 'antialias' in kwargs:
        del kwargs['antialias']
    return _original_interpolate(*args, **kwargs)
F.interpolate = _patched_interpolate

if not hasattr(F, 'scaled_dot_product_attention'):
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        
        if is_causal:
            L, S = query.size(-2), key.size(-2)
            causal_mask = torch.ones((L, S), dtype=torch.bool, device=query.device).tril()
            attn_weight.masked_fill_(~causal_mask, float('-inf'))
            
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weight.masked_fill_(~attn_mask, float('-inf'))
            else:
                attn_weight += attn_mask
                
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        if dropout_p > 0.0:
            attn_weight = F.dropout(attn_weight, p=dropout_p)
            
        return torch.matmul(attn_weight, value)
        
    F.scaled_dot_product_attention = scaled_dot_product_attention




def crop_local_window(image, mask, crop_size=518):
    rows, columns = np.where(mask > 0)
    if len(rows) == 0:
        return image[:crop_size, :crop_size], 0, 0

    center_row = int(np.mean(rows))
    center_column = int(np.mean(columns))
    half = crop_size // 2
    height, width = image.shape[:2]
    top, bottom = center_row - half, center_row + half
    left, right = center_column - half, center_column + half
    pad_top, pad_bottom = max(0, -top), max(0, bottom - height)
    pad_left, pad_right = max(0, -left), max(0, right - width)
    valid_top, valid_bottom = max(0, top), min(height, bottom)
    valid_left, valid_right = max(0, left), min(width, right)
    crop = image[valid_top:valid_bottom, valid_left:valid_right]
    if pad_top or pad_bottom or pad_left or pad_right:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return crop, valid_top - pad_top, valid_left - pad_left


class DINOv2RGBEncoder(nn.Module):
    def __init__(self, dino_size='vits14'):
        """Initialize a DINOv2 encoder from the local repository."""
        super().__init__()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        repo_dir = os.path.join(parent_dir, 'dinov2_local')

        self.backbone = torch.hub.load(repo_dir, f'dinov2_{dino_size}', source='local', pretrained=True)
        print("Successfully loaded DINOv2 model (weights automatically managed by torch hub)")
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.out_channels = self.backbone.embed_dim 
        self.patch_size = 14
    

    def forward(self, x):
        """Encode an RGB tensor with shape (B, 3, H, W)."""
        B, C, H, W = x.shape
        
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        
        if pad_w > 0 or pad_h > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        H_pad, W_pad = x.shape[2:] 
        h_featmap = H_pad // self.patch_size
        w_featmap = W_pad // self.patch_size

        features = self.backbone.forward_features(x)
        patch_tokens = features['x_norm_patchtokens'] 

        spatial_features = patch_tokens.permute(0, 2, 1).reshape(B, self.out_channels, h_featmap, w_featmap)

        dense_features = F.interpolate(spatial_features, size=(H_pad, W_pad), mode='bilinear', align_corners=False)
        
        if pad_w > 0 or pad_h > 0:
            dense_features = dense_features[:, :, :H, :W]

        return dense_features