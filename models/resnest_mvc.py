import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
from timm.models.resnet import ResNet
from timm.models.resnest import ResNestBottleneck
import torch.nn as nn
import segmentation_models_pytorch as smp

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
    return (torch.exp(-torch.abs(x - 0.5)) - torch.exp(torch.tensor(-1 / 2))) + 1


class VCModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()


        self.general_conv = VGGBlock(in_channels, in_channels // 4, kernel_size, stride, padding)
        print()

        self.conv_top1 = VGGBlock(in_channels // 4, in_channels // 8, kernel_size, padding, stride)
        self.conv_top2 = VGGBlock(in_channels // 8, in_channels // 8, kernel_size, padding, stride)
        self.conv_top11 = nn.Conv2d(in_channels // 8, 1, 1, padding=0, stride=stride)

        self.conv_bot1 = VGGBlock(in_channels // 4, in_channels // 8, kernel_size, padding, stride)
        self.conv_bot2 = VGGBlock(in_channels // 8, in_channels // 8, kernel_size, padding, stride)

        self.end_conv1 = nn.Conv2d(in_channels // 8, out_channels, 1, padding=0, stride=stride)

    def forward(self, x2):
        x2 = self.general_conv(x2)
        x1 = self.conv_top1(x2)
        x1 = self.conv_top2(x1)
        x1 = self.conv_top11(x1).sigmoid()  # channel = 1

        x2 = self.conv_bot1(x2)
        x2 = self.conv_bot2(x2)  # channel = 32

        # Matrix Multiplication x1 * x2
        x2_x1 = x2 * x1
        x1 = F_0(x1)
        x2_x1_x1 = x2_x1 * x1
        return self.end_conv1(x2_x1_x1)

#
# if __name__ == "__main__":
#     model = VCModule(128, 1)
#     x = torch.randn(1, 128, 256, 256)
#     y = model(x)
#     print(y.shape)
#     print(y)

class ResNestEncoderMVC(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        self.mvcs_layers = nn.ModuleList([nn.Identity()])
        for i in self.out_channels[1:]:
            # Keep Out Channels = in
            self.mvcs_layers.append(VCModule(i, i))

        del self.fc
        del self.global_pool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.act1),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def make_dilated(self, *args, **kwargs):
        raise ValueError("ResNest encoders do not support dilated mode")

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = self.mvcs_layers[i](stages[i](x))
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)


resnest_weights = {
    "timm-resnest14d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth",
        # noqa
    },
    "timm-resnest26d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth",
        # noqa
    },
    "timm-resnest50d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth",
        # noqa
    },
    "timm-resnest101e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth",
        # noqa
    },
    "timm-resnest200e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth",
        # noqa
    },
    "timm-resnest269e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth",
        # noqa
    },
    "timm-resnest50d_4s2x40d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_4s2x40d-41d14ed0.pth",
        # noqa
    },
    "timm-resnest50d_1s4x24d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_1s4x24d-d4a4f76f.pth",
        # noqa
    },
}

pretrained_settings = {}
for model_name, sources in resnest_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }

timm_resnest_encoders = {
    "timm-resnest14d-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest14d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [1, 1, 1, 1],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest26d-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest26d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [2, 2, 2, 2],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest50d-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest50d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest101e-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest101e"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [3, 4, 23, 3],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest200e-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest200e"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [3, 24, 36, 3],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest269e-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest269e"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [3, 30, 48, 8],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest50d_4s2x40d-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest50d_4s2x40d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 40,
            "cardinality": 2,
            "block_args": {"radix": 4, "avd": True, "avd_first": True},
        },
    },
    "timm-resnest50d_1s4x24d-mvc": {
        "encoder": ResNestEncoderMVC,
        "pretrained_settings": pretrained_settings["timm-resnest50d_1s4x24d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNestBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 24,
            "cardinality": 4,
            "block_args": {"radix": 1, "avd": True, "avd_first": True},
        },
    },
}


def init_resnest_mvc_to_smp():
    for name, encoder in timm_resnest_encoders.items():
        smp.encoders.encoders[name] = encoder
        print(f"Added new Model:\t{name}")


if __name__ == '__main__':
    # to mac os
    init_resnest_mvc_to_smp()
    unetpp = smp.UnetPlusPlus(encoder_name='timm-resnest50d-mvc', encoder_weights=None, classes=1, in_channels=3).to('mps')
    a = torch.randn(4, 3, 512, 512).to('mps')
    b = unetpp(a)
    print(b.shape)

