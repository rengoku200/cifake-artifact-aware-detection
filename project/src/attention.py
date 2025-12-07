import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention module.
    Learns which channels (feature maps) are more important.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels)
        )



    def forward(self, x):
        b, c, _, _ = x.size()

        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)

        attn_avg = self.mlp(avg)
        attn_max = self.mlp(mx)

        attn = torch.sigmoid(attn_avg + attn_max).view(b, c, 1, 1)
        return x * attn
    
class SpatialAttention(nn.Module):
    
    """
    Spatial attention module.
    Learns which spatial locations are more important.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
    
    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn
    
class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention(kernel_size=7)
    
    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
