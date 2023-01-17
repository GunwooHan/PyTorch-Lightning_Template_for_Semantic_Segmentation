""" ResNeSt Models

Paper: `ResNeSt: Split-Attention Networks` - https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl w/ weights at https://github.com/zhanghang1989/ResNeSt by Hang Zhang

Modified for torchscript compat, and consistency with timm by Ross Wightman
"""
from losses import FocalLoss
import inspect
import sys
import os
from torchmetrics.functional import jaccard_index, f1_score, precision, recall, accuracy
from adamp import AdamP
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pyrsistent import m
from base64 import encode
from timm.models.layers import make_divisible
import math

import torch
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from timm.models.resnet import ResNet
import torch.nn as nn
import torch.nn.functional as F
from . import MPNCOV
from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch as smp

""" Split Attention Conv2d (for ResNeSt Models)

Paper: `ResNeSt: Split-Attention Networks` - /https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl at https://github.com/zhanghang1989/ResNeSt

Modified for torchscript compat, performance, and consistency with timm by Ross Wightman
"""


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    """Split-Attention (aka Splat)
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.ReLU, norm_layer=None, drop_layer=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer(inplace=False)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer(inplace=False)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix,
                   RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1.0', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'resnest14d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth'),
    'resnest26d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth'),
    'resnest50d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth'),
    'resnest101e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'resnest200e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=0.909, interpolation='bicubic'),
    'resnest269e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth',
        input_size=(3, 416, 416), pool_size=(13, 13), crop_pct=0.928, interpolation='bicubic'),
    'resnest50d_4s2x40d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_4s2x40d-41d14ed0.pth',
        interpolation='bicubic'),
    'resnest50d_1s4x24d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_1s4x24d-d4a4f76f.pth',
        interpolation='bicubic')
}


def cov_feature(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h * w
    x = x.reshape(batchsize, dim, M)
    I_hat = (-1. / M / M) * torch.ones(dim, dim, device=x.device) + \
        (1. / M) * torch.eye(dim, dim, device=x.device)
    I_hat = I_hat.view(1, dim, dim).repeat(batchsize, 1, 1).type(x.dtype)
    y = (x.transpose(1, 2)).bmm(I_hat).bmm(x)
    return y


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None,
            radix=1, cardinality=1, base_width=64, avd=False, avd_first=False, is_first=False,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, attention='+', att_dim=128):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv2d(inplanes, group_width,
                               kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=False)
        self.avd_first = nn.AvgPool2d(
            3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None

        if self.radix >= 1:
            self.conv2 = SplitAttn(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_layer=drop_block)
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer(inplace=False)
        self.avd_last = nn.AvgPool2d(
            3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None

        self.conv3 = nn.Conv2d(group_width, planes * 4,
                               kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.act3 = act_layer(inplace=False)

        if attention in {'1', '+', 'M', '&'}:
            if planes > 64:
                DR_stride = 1
            else:
                DR_stride = 2

            self.ch_dim = att_dim
            self.conv_for_DR = nn.Conv2d(
                planes * self.expansion, self.ch_dim,
                kernel_size=1, stride=DR_stride, bias=True)
            self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
            self.row_bn = nn.BatchNorm2d(self.ch_dim)
            # row-wise conv is realized by group conv
            self.row_conv_group = nn.Conv2d(
                self.ch_dim, 4 * self.ch_dim,
                kernel_size=(self.ch_dim, 1),
                groups=self.ch_dim, bias=True)
            self.fc_adapt_channels = nn.Conv2d(
                4 * self.ch_dim, planes * self.expansion,
                kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()

        if attention in {'2', '+', 'M', '&'}:
            self.sp_d = att_dim
            self.sp_h = 8
            self.sp_w = 8
            self.sp_reso = self.sp_h * self.sp_w
            self.conv_for_DR_spatial = nn.Conv2d(
                planes * self.expansion, self.sp_d,
                kernel_size=1, stride=1, bias=True)
            self.bn_for_DR_spatial = nn.BatchNorm2d(self.sp_d)

            self.adppool = nn.AdaptiveAvgPool2d((self.sp_h, self.sp_w))
            self.row_bn_for_spatial = nn.BatchNorm2d(self.sp_reso)
            # row-wise conv is realized by group conv
            self.row_conv_group_for_spatial = nn.Conv2d(
                self.sp_reso, self.sp_reso * 4, kernel_size=(self.sp_reso, 1),
                groups=self.sp_reso, bias=True)
            self.fc_adapt_channels_for_spatial = nn.Conv2d(
                self.sp_reso * 4, self.sp_reso, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()
            self.adpunpool = F.adaptive_avg_pool2d
            self.relu = nn.ReLU(inplace=False)
            self.relu_normal = nn.ReLU(inplace=False)

        if attention == '&':  # we employ a weighted spatial concat to keep dim
            self.groups_base = 32
            self.groups = int(planes * self.expansion / 64)
            self.factor = int(math.log(self.groups_base / self.groups, 2))
            self.padding_num = self.factor + 2
            self.conv_kernel_size = self.factor * 2 + 5
            self.dilate_conv_for_concat1 = nn.Conv2d(planes * self.expansion,
                                                     planes * self.expansion,
                                                     kernel_size=(
                                                         self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.dilate_conv_for_concat2 = nn.Conv2d(planes * self.expansion,
                                                     planes * self.expansion,
                                                     kernel_size=(
                                                         self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.bn_for_concat = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.attention = attention

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def chan_att(self, out):
        # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR(out)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = MPNCOV.CovpoolLayer(out)  # Nxdxd
        out = out.view(out.size(0), out.size(
            1), out.size(2), 1).contiguous()  # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out)  # Nx512x1x1

        out = self.fc_adapt_channels(out)  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1

        return out

    def pos_att(self, out):
        pre_att = out  # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR_spatial(out)
        out = self.bn_for_DR_spatial(out)

        out = self.adppool(out)  # keep the feature map size to 8x8

        out = cov_feature(out)  # Nx64x64
        out = out.view(out.size(0), out.size(1), out.size(2),
                       1).contiguous()  # Nx64x64x1
        out = self.row_bn_for_spatial(out)

        out = self.row_conv_group_for_spatial(out)  # Nx256x1x1
        out = self.relu(out)

        out = self.fc_adapt_channels_for_spatial(out)  # Nx64x1x1
        out = self.sigmoid(out)
        out = out.view(out.size(0), 1, self.sp_h,
                       self.sp_w).contiguous()  # Nx1x8x8

        out = self.adpunpool(
            out, (pre_att.size(2), pre_att.size(3)))  # unpool Nx1xHxW

        return out

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        if self.attention == '1':  # channel attention,GSoP default mode
            pre_att = out
            att = self.chan_att(out)
            out = pre_att * att

        elif self.attention == '2':  # position attention
            pre_att = out
            att = self.pos_att(out)
            out = self.relu_normal(pre_att * att)

        elif self.attention == '+':  # fusion manner: average
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = pre_att * chan_att + self.relu(pre_att.clone() * pos_att)

        elif self.attention == 'M':  # fusion manner: MAX
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = torch.max(pre_att * chan_att,
                            self.relu(pre_att.clone() * pos_att))

        elif self.attention == '&':  # fusion manner: concat
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out1 = self.dilate_conv_for_concat1(pre_att * chan_att)
            out2 = self.dilate_conv_for_concat2(self.relu(pre_att * pos_att))
            out = out1 + out2
            out = self.bn_for_concat(out)

        out_o = out + shortcut
        out_o = self.act3(out_o)
        return out_o


def _create_resnest(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model
def resnest14d(pretrained=False, **kwargs):
    """ ResNeSt-14d model. Weights ported from GluonCV.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[1, 1, 1, 1],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False), **kwargs)
    return _create_resnest('resnest14d', pretrained=pretrained, **model_kwargs)


@register_model
def resnest26d(pretrained=False, **kwargs):
    """ ResNeSt-26d model. Weights ported from GluonCV.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[2, 2, 2, 2],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False), **kwargs)
    return _create_resnest('resnest26d', pretrained=pretrained, **model_kwargs)


@register_model
def resnest50d(pretrained=False, **kwargs):
    """ ResNeSt-50d model. Matches paper ResNeSt-50 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'd' for deep stem, stem_width 32, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False), **kwargs)
    return _create_resnest('resnest50d', pretrained=pretrained, **model_kwargs)


@register_model
def resnest101e(pretrained=False, **kwargs):
    """ ResNeSt-101e model. Matches paper ResNeSt-101 model, https://arxiv.org/abs/2004.08955
     Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 23, 3],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False), **kwargs)
    return _create_resnest('resnest101e', pretrained=pretrained, **model_kwargs)


@register_model
def resnest200e(pretrained=False, **kwargs):
    """ ResNeSt-200e model. Matches paper ResNeSt-200 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 24, 36, 3],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False), **kwargs)
    return _create_resnest('resnest200e', pretrained=pretrained, **model_kwargs)


@register_model
def resnest269e(pretrained=False, **kwargs):
    """ ResNeSt-269e model. Matches paper ResNeSt-269 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 30, 48, 8],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False), **kwargs)
    return _create_resnest('resnest269e', pretrained=pretrained, **model_kwargs)


@register_model
def resnest50d_4s2x40d(pretrained=False, **kwargs):
    """ResNeSt-50 4s2x40d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=40, cardinality=2,
        block_args=dict(radix=4, avd=True, avd_first=True), **kwargs)
    return _create_resnest('resnest50d_4s2x40d', pretrained=pretrained, **model_kwargs)


@register_model
def resnest50d_1s4x24d(pretrained=False, **kwargs):
    """ResNeSt-50 1s4x24d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=24, cardinality=4,
        block_args=dict(radix=1, avd=True, avd_first=True), **kwargs)
    return _create_resnest('resnest50d_1s4x24d', pretrained=pretrained, **model_kwargs)


# ________________ Register Module To smp ______________________

class ResNestEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

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
    "timm-resnest14d": {
        "encoder": ResNestEncoder,
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
    "timm-resnest26d": {
        "encoder": ResNestEncoder,
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
    "timm-resnest50d": {
        "encoder": ResNestEncoder,
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
    "timm-resnest101e": {
        "encoder": ResNestEncoder,
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
    "timm-resnest200e": {
        "encoder": ResNestEncoder,
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
    "timm-resnest269e": {
        "encoder": ResNestEncoder,
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
    "timm-resnest50d_4s2x40d": {
        "encoder": ResNestEncoder,
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
    "timm-resnest50d_1s4x24d": {
        "encoder": ResNestEncoder,
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


for k, v in timm_resnest_encoders.items():
    smp.encoders.encoders[f'{k}-meangsop'] = v
    print(f'Added Model:\t{k}-meangsop')


sys.path.append(os.path.realpath(os.path.join(
    os.path.dirname(inspect.getfile(inspect.currentframe())), '../')))


""" Parts of the U-Net model """


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()


def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    # dice = dice_fn(y_pred.sigmoid(), y_true)
    return bce
    # return bce

# n_loss = FocalLoss(gamma=2)
# n_loss = nn.NLLLoss()


# def loss_fn(outputs, targets):
#     return  n_loss(outputs, targets)


def mask_onehot(masks):
    # batch_size, h, w
    masks = masks.long()
    masks_onehot = torch.zeros(masks.size(0), masks.max(
    ) + 1, masks.size(1), masks.size(2)).to(masks.device)
    masks_onehot = masks_onehot.scatter_(1, masks.unsqueeze(1), 1)
    return masks_onehot


class ResNeStGSoPUPnetPPModel(pl.LightningModule):
    def __init__(self, args=None, encoder='timm-resnest26d-addgsop'):
        super().__init__()
        # 取消预训练
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
        self.args = args
        self.criterion = loss_fn

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args.learning_rate, eps=1e-5)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.args.learning_rate, eps=1e-5)
        elif self.args.optimizer == 'adamp':
            optimizer = AdamP(self.parameters(), lr=self.args.learning_rate, betas=(
                0.9, 0.999), weight_decay=1e-4, eps=1e-5)
        elif self.args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(
                self.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        elif self.args.optimizer == 'asgd':
            optimizer = torch.optim.ASGD(
                self.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        

        if self.args.scheduler == "reducelr":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, mode="max", verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/jac_idx"}

        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        # Mask 增加一个维度
        mask = mask.long()

        outputs = self.model(image)
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        pre_label = outputs.sigmoid()
        jaccard_index_value = jaccard_index(pre_label, mask, num_classes=2)
        f1 = f1_score(pre_label, mask, multiclass=True,
                      num_classes=2, mdmc_average='global')
        acc = accuracy(pre_label, mask, multiclass=True,
                       num_classes=2, mdmc_average='global')

        self.log('train/loss', loss, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jac_idx', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('train/f1', f1, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/acc', acc, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "jac_idx": jaccard_index_value, "f1": f1, "acc": acc}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        # Mask 增加一个维度
        mask = mask.long()

        outputs = self.model(image)
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        pre_label = outputs.sigmoid()
        jaccard_index_value = jaccard_index(pre_label, mask, num_classes=2)
        f1 = f1_score(pre_label, mask, multiclass=True,
                      num_classes=2, mdmc_average='global')
        acc = accuracy(pre_label, mask, multiclass=True,
                       num_classes=2, mdmc_average='global')

        self.log('val/loss', loss, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jac_idx', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('val/f1', f1, on_epoch=True, on_step=True,
                 prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "jac_idx": jaccard_index_value, "f1": f1, "acc": acc}

    def test_step(self, test_batch, batch_idx):
        image, mask = test_batch
        mask = mask.long()

        outputs = self.model(image)
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        pre_label = outputs.sigmoid()
        jaccard_index_value = jaccard_index(pre_label, mask, num_classes=2)
        f1 = f1_score(pre_label, mask, multiclass=True,
                      num_classes=2, mdmc_average='global')
        acc = accuracy(pre_label, mask, multiclass=True,
                       num_classes=2, mdmc_average='global')

        self.log('test/loss', loss, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)
        self.log('test/jac_idx', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('test/f1', f1, on_epoch=True, on_step=True,
                 prog_bar=True, sync_dist=True)
        self.log('test/acc', acc, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "jac_idx": jaccard_index_value, "f1": f1, "acc": acc}

# import segmentation_models_pytorch as smp


if __name__ == '__main__':
    model = ResNeStGSoPUPnetPPModel()
    print(model)
