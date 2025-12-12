import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAKECNN(nn.Module):
    """
    Reimplment CNN architecture from the paper (need to reshape the image to (32*32))
    """
    def __init__(self):
        super().__init__()
        

        # Conv2D → 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  

        # After conv1+pool: 32x16x16
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 2)  

    def forward(self, x):
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))  

        x = x.view(x.size(0), -1)           
        x = F.relu(self.fc1(x))              
        logits = self.fc2(x)                 
        return logits