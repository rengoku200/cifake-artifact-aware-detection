import torch
import torch.nn as nn
import torch.nn.functional as F



class FreqBranch(nn.Module):
  """ Frequency-domain branch.
    Converts RGB image to frequency magnitude via FFT and then
    runs a small CNN over that representation.

    Input:
        x: [B, 3, H, W] RGB images
    Output:
        feature map, e.g. [B, 64, 16, 16] for H=W=128
    """
  
  def __init__(self,in_channels=3, base_channels = 32):
    super().__init__()
    
    self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2,2)


  def forward(self, x):

    # 1) FFT: go to frequency domain (complex numbers)
    freq = torch.fft.fft2(x) 
    freq = torch.fft.fftshift(freq, dim=(-2, -1)) 

    #2) Magnitude and log for stability
    mag = torch.log1p(torch.abs(freq))

    #3) CNN ove frequency magnitude 
    x = self.pool(F.relu(self.conv1(mag)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))

    return x 
  

  



