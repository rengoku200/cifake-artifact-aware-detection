import torch  
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Simple CNN for binary classification: REAL (0) vs FAKE (1).
    Input: [B, 3, 128, 128]
    Output: logits [B, 2]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2) 

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  

        x = x.view(x.size(0), -1)      
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                    
        return x
