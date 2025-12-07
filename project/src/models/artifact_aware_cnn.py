import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CBAMBlock
from freq_model import FreqBranch



class SpatialBranch(nn.Module):
    """ Spatial-domain branch.
      CNN with CBAM attention blocks.
      Output shape matches FrequencyAttention output.
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

        return x

class ArtifactAwareCNN(nn.Module):
    """
    Our upgraded architecture:

    - SpatialBranch processes the RGB image.
    - FrequencyBranch (from freq_model.py) processes FFT magnitude.
    - We concatenate both feature maps along the channel dimension.
    - CBAMBlock (from attention.py) applies channel+spatial attention.
    - Global pooling + MLP classify REAL vs FAKE.

    Input:
        x: [B, 3, 128, 128]
    Output:
        logits: [B, 2]  (class 0 = REAL, class 1 = FAKE)
    """


    def __init__(self, num_classes =2, base_channels=32):
        super().__init__()

        self.spatial_branch = SpatialBranch(in_channels=3, base_channels=base_channels)
        self.freq_branch = FreqBranch(in_channels=3, base_channels=base_channels)

        # Each branch ends with C = base_channels * 2
        # so concatenation doubles the channels.
        fused_channels = base_channels*4 

        self.cbam_fusion = CBAMBlock(fused_channels)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(fused_channels, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        #Spatial features 
        spatial_feat = self.spatial_branch(x)
        #Frequency featues
        freq_feat = self.freq_branch(x)

        # fuse along channel and spatial dimensions
        fused = torch.cat([spatial_feat, freq_feat], dim=1)

        #CBAM attention
        attended = self.cbam_fusion(fused)

        # Global pooling
        pooled = self.global_pool(attended)  
        pooled = pooled.view(pooled.size(0), -1)  

        # MLP classifier
        x = F.relu(self.fc1(pooled))
        logits = self.fc2(x)

        return logits
