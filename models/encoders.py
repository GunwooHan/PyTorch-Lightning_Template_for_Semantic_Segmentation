# Define Some Segmentation Encoders
# Author: AiEson

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from typing import Optional
from functools import partial

class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x
    
class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )
        
Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)

class SE (nn.Module):
    def __init__(self, in_features: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid(),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# EfficientNet V2 MBConvBlock
class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features, 
                                      expanded_features, 
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # SE Module
                        SE(expanded_features),
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )
    
    
if __name__ == "__main__":
    # model = EfficientNetV2_UNetPlusPlus(3, 1)
    # x = torch.randn(1, 3, 256, 256)
    # y = model(x)
    # print(y.shape)