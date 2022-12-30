import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(VGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
def F_0(x):
    # alpha * (e^(-abs(x - 0.5)) - e^(-0.5)) + 1
    return (torch.exp(-torch.abs(x - 0.5)) - torch.exp(torch.tensor(-1/2)))+ 1


class VCModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        
        self.general_conv = VGGBlock(in_channels, 64, kernel_size, stride, padding)
        
        self.conv_top1 = VGGBlock(64, 32, kernel_size, padding, stride)
        self.conv_top2 = VGGBlock(32, 32, kernel_size, padding, stride)
        self.conv_top11 = nn.Conv2d(32, 1, 1, padding=0, stride=stride)
        
        self.conv_bot1 = VGGBlock(64, 32, kernel_size, padding, stride)
        self.conv_bot2 = VGGBlock(32, 32, kernel_size, padding, stride)
        
        self.end_conv1 = nn.Conv2d(32, out_channels, 1, padding=0, stride=stride)
    
    def forward(self, x2):
        x2 = self.general_conv(x2)
        x1 = self.conv_top1(x2)
        x1 = self.conv_top2(x1)
        x1 = self.conv_top11(x1).sigmoid() # channel = 1
        
        x2 = self.conv_bot1(x2)
        x2 = self.conv_bot2(x2) # channel = 32
        
        # Matrix Multiplication x1 * x2
        x2_x1 = x2 * x1
        x1 = F_0(x1)
        x2_x1_x1 = x2_x1 * x1
        return self.end_conv1(x2_x1_x1)
    
    
if __name__ == "__main__":
    model = VCModule(128, 1)
    x = torch.randn(1, 128, 256, 256)
    y = model(x)
    print(y.shape)
    print(y)