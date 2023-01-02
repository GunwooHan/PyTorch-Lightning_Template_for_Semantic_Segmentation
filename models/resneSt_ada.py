import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
from timm.models.resnet import ResNet
from timm.models.resnest import ResNestBottleneck
import torch.nn as nn
import adapool_cuda
from adaPool import AdaPool1d, AdaPool2d, AdaPool3d
import segmentation_models_pytorch as smp


class ResNestEncoderAdaPool(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        # Replace AvgPool2d with MaxPool2d
        self.rep_pool33 = AdaPool2d(kernel_size=2, stride=2, beta=(1, 1))
        self.rep_pool22 = AdaPool2d(kernel_size=2, stride=2, beta=(1, 1))

        if self.maxpool is not None:
            setattr(self, 'maxpool', self.rep_pool33)

        for layer_i in range(4):
            r_i = layer_i + 1
            for i in range(len(getattr(self, f'layer{r_i}'))):
                # print(self.layer2[i])
                if hasattr(eval(f'self.layer{r_i}[i]'), 'avd_last'):
                    if eval(f'self.layer{r_i}[i]').avd_last is not None:
                        eval(f'self.layer{r_i}[i]').avd_last = self.rep_pool33
                if hasattr(eval(f'self.layer{r_i}[i]'), 'downsample'):
                    if eval(f'self.layer{r_i}[i]').downsample is not None:
                        print(eval(f'self.layer{r_i}[i]').downsample[0])
                        if not isinstance(eval(f'self.layer{r_i}[i]').downsample[0], nn.Identity):
                            eval(f'self.layer{r_i}[i]').downsample[0] = self.rep_pool22

        del self.rep_pool33
        del self.rep_pool22
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
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)

resnest_weights = {
    "timm-resnest14d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth",  # noqa
    },
    "timm-resnest26d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth",  # noqa
    },
    "timm-resnest50d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth",  # noqa
    },
    "timm-resnest101e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth",  # noqa
    },
    "timm-resnest200e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth",  # noqa
    },
    "timm-resnest269e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth",  # noqa
    },
    "timm-resnest50d_4s2x40d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_4s2x40d-41d14ed0.pth",  # noqa
    },
    "timm-resnest50d_1s4x24d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_1s4x24d-d4a4f76f.pth",  # noqa
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
    "timm-resnest14d-ada": {
        "encoder": ResNestEncoderAdaPool,
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
    "timm-resnest26d-ada": {
        "encoder": ResNestEncoderAdaPool,
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
    "timm-resnest50d-ada": {
        "encoder": ResNestEncoderAdaPool,
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
    "timm-resnest101e-ada": {
        "encoder": ResNestEncoderAdaPool,
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
    "timm-resnest200e-ada": {
        "encoder": ResNestEncoderAdaPool,
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
    "timm-resnest269e-ada": {
        "encoder": ResNestEncoderAdaPool,
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
    "timm-resnest50d_4s2x40d-ada": {
        "encoder": ResNestEncoderAdaPool,
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
    "timm-resnest50d_1s4x24d-ada": {
        "encoder": ResNestEncoderAdaPool,
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


def init_resnest_ada_to_smp():
    for name, encoder in timm_resnest_encoders.items():
        smp.encoders.encoders[name] = encoder
        print(f"Added new Model:\t{name}")

if __name__ == '__main__':
    unetpp = smp.UnetPlusPlus(encoder_name='timm-resnest50d-ada', encoder_weights=None, classes=1, in_channels=3).to('cuda')
    a = torch.randn(4, 3, 512, 512).to('cuda')
    b = unetpp(a)
    print(b.shape)
    
    