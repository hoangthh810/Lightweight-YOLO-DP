# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode
from utils.EfficientAttention import EfficientAttention
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# class DecoupledHead(nn.Module):
#     def __init__(self, ch=256, nc=80,  anchors=()):
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.merge = Conv(ch, 256 , 1, 1)
#         self.cls_convs1 = Conv(256 , 256 , 3, 1, 1)
#         self.cls_convs2 = Conv(256 , 256 , 3, 1, 1)
#         self.reg_convs1 = Conv(256 , 256 , 3, 1, 1)
#         self.reg_convs2 = Conv(256 , 256 , 3, 1, 1)
#         self.cls_preds = nn.Conv2d(256 , self.nc * self.na, 1) # 一个1x1的卷积，把通道数变成类别数，比如coco 80类（主要对目标框的类别，预测分数）
#         self.reg_preds = nn.Conv2d(256 , 4 * self.na, 1)       # 一个1x1的卷积，把通道数变成4通道，因为位置是xywh
#         self.obj_preds = nn.Conv2d(256 , 1 * self.na, 1)       # 一个1x1的卷积，把通道数变成1通道，通过一个值即可判断有无目标（置信度）
#
#     def forward(self, x):
#         x = self.merge(x)
#         x1 = self.cls_convs1(x)
#         x1 = self.cls_convs2(x1)
#         x1 = self.cls_preds(x1)
#         x2 = self.reg_convs1(x)
#         x2 = self.reg_convs2(x2)
#         x21 = self.reg_preds(x2)
#         x22 = self.obj_preds(x2)
#         out = torch.cat([x21, x22, x1], 1) # 把分类和回归结果按channel维度，即dim=1拼接
#         return out

# FIX
class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super().__init__()
        self.nc = nc
        self.na = len(anchors[0]) // 2

        # Dùng ch thay vì fixed 256
        c_ = max(ch // 2, 64)  # adaptive, tối thiểu 64

        self.merge      = Conv(ch, c_, 1, 1)
        self.cls_convs  = Conv(c_, c_, 3, 1, 1)   # 1 layer thay vì 2
        self.reg_convs  = Conv(c_, c_, 3, 1, 1)   # 1 layer thay vì 2
        self.cls_preds  = nn.Conv2d(c_, self.nc * self.na, 1)
        self.reg_preds  = nn.Conv2d(c_, 4 * self.na, 1)
        self.obj_preds  = nn.Conv2d(c_, 1 * self.na, 1)

    def forward(self, x):
        x  = self.merge(x)
        x1 = self.cls_preds(self.cls_convs(x))
        x2 = self.reg_convs(x)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        return torch.cat([x21, x22, x1], 1)


class Decoupled_Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode
 
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
 
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(DecoupledHead(x, nc, anchors) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
 
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
 
            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
 
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
 
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
 
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
    # default_act = nn.ReLU6(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class FAMBlock(nn.Module):
    def __init__(self, channel, gamma=2):
        super(FAMBlock,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // gamma, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // gamma, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class Cfc(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Cfc,self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(4 * out_channel, out_channel, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.silu = nn.SiLU()
        # self.silu = nn.ReLU6(inplace=True)
        self.FAM = FAMBlock(4 * out_channel)

    def forward(self, x):
        y1 = self.bn(self.conv1(x))
        y2 = self.silu(self.bn(self.conv2(y1)))
        y3 = self.silu(self.bn(self.conv2(y1 + y2)))
        y4 = self.silu(self.bn(self.conv2(y1 + y3)))
        y = torch.cat([y1, y2, y3, y4], 1)
        return self.conv3(self.FAM(y))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        # self.act = nn.ReLU6(inplace=True)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class LSKmodule(nn.Module):
    def __init__(self, c1,dim):
        super().__init__()
        self.c1=c1
        self.dim=dim
        # self.s = s
        self.conv0 = nn.Conv2d(c1, dim, 5, padding=2, groups=dim)
        # self.conv0 = GhostConv1(c1, dim, 5, p=2, g=int(dim/2))
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # self.convl = GhostConv1(dim, dim, 7, s=1, p=9, g=int(dim/2), d=3)
        # self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv0_s = GhostConv(dim, dim // 2, 1)
        # self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = GhostConv(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # self.conv_m = nn.Conv2d(dim // 2, dim, 1)
        self.conv_m = GhostConv(dim // 2, dim, 1)

    def forward(self, x):
        attn0 = self.conv0(x)
        # print(attn0.shape)
        attn2 = self.convl(attn0)

        attn1 = self.conv0_s(attn0)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)
        # if self.s == 1:
        #     # print(attn1.shape, attn.shape,self.c1,self.dim)
        #     return attn0 * attn
        # elif self.s == 0:
        return x * attn

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.L = LSKmodule(c1, c_, 1)
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3g(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def backward_hook(n,module, grad_in, grad_out):
        return  grad_out

    def forward(self, x):
        z = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        # y = self.register_backward_hook(self.backward_hook)
        print(x.grad_fn)
        return z


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')
        elif paddle:  # PaddlePaddle
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix('.pdiparams')
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f'image{i}'  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                display(im) if is_notebook() else im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


class C2f(nn.Module):
    """C2f: YOLOv8-style Cross-Stage Partial với multi-output concat.
    Khác C3: concat TẤT CẢ intermediate bottleneck output -> gradient flow tốt hơn.
    Dùng cho head YOLO-DP khi task cần giữ detail nhỏ (PDT) hoặc fine-grained texture (CWC).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # split -> [chunk1, chunk2]
        y.extend(m(y[-1]) for m in self.m)  # mỗi bottleneck nối tiếp chunk cuối
        return self.cv2(torch.cat(y, 1))  # concat tất cả


class C2f_Ghost(nn.Module):
    """C2f_Ghost: thay Bottleneck bằng GhostBottleneck bên trong C2f.
    GhostConv ở cả cv1/cv2 lẫn bottleneck -> giảm ~40% params so với C2f.
    LƯU Ý: Neck YOLO-DP đã dùng GhostConv -> double compression.
    Phù hợp CWC (large object) hơn PDT (small dense target).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = GhostConv(c1, 2 * self.c, 1, 1)
        self.cv2 = GhostConv((2 + n) * self.c, c2, 1)
        # GhostBottleneck(c1, c2, k, s): s=1 -> identity shortcut tự động
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, self.c) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class LSKAttention(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        # Nhánh 1: Kernel 5x5 thông thường
        self.conv5 = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)
        # Nhánh 2: Kernel 7x7 với dilation=3 (tương đương vùng nhìn 23x23)
        self.conv7 = nn.Conv2d(c1, c1, 7, stride=1, padding=9, dilation=3, groups=c1)

        # Bộ chọn lọc không gian (Spatial Selection)
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),  # Paper dùng kernel lớn để bắt bối cảnh
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(c1, c1, 1)  # Channel mixing
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        # self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        # 1. Thu thập đặc trưng song song (Parallel decomposition)
        feat1 = self.conv5(x)
        feat2 = self.conv7(x)

        # 2. Tính toán Attention map dựa trên Max và Avg pooling (theo channel)
        # Gom thông tin từ cả 2 scale
        feats = feat1 + feat2
        attn_avg = torch.mean(feats, dim=1, keepdim=True)
        attn_max, _ = torch.max(feats, dim=1, keepdim=True)

        # Trọng số lựa chọn vùng nhìn
        attn = torch.cat([attn_avg, attn_max], dim=1)
        attn = self.conv_spatial(attn)

        # 3. Kết hợp và hòa trộn kênh
        out = feat1 * attn + feat2 * (1 - attn)  # Chọn lọc động giữa 2 scale
        out = self.conv1(out)

        return x + out

class LowRankConv1x1(nn.Module):
    """
    Thay Conv 1x1 (Cin->Cout) bằng 2 conv nhỏ: Cin->r->Cout
    + residual shortcut khi Cin==Cout (theo bài báo section III.C)
    """
    def __init__(self, c1, c2, r=None, act=True):
        super().__init__()
        if r is None:
            r = max(c1 // 4, 16)  # rank mặc định
        self.reduce = Conv(c1, r, 1, 1, act=act)   # Cin → r
        self.expand = Conv(r, c2, 1, 1, act=act)   # r → Cout
        self.shortcut = (c1 == c2)                  # chỉ add khi dims khớp

    def forward(self, x):
        out = self.expand(self.reduce(x))
        return x + out if self.shortcut else out

class DSConv(nn.Module):
    """
    Depthwise Separable Conv: depthwise k×k + pointwise LowRank 1×1
    Theo bài báo: DSC + LR kết hợp trên pointwise layer
    """
    def __init__(self, c1, c2, k=3, s=1, r=None, act=True):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1, act=act)        # depthwise
        self.pw = LowRankConv1x1(c1, c2, r=r, act=act)     # pointwise LR

    def forward(self, x):
        return self.pw(self.dw(x))

class SPPF_LR(nn.Module):
    """
    SPPF tối ưu: thay 2 Conv 1×1 bằng LowRankConv1x1
    cv1: c1 → c_  (c1 != c_ nên không có shortcut)
    cv2: c_*4 → c2 (không có shortcut)
    MaxPool giữ nguyên - không có params
    """
    def __init__(self, c1, c2, k=5, r=None):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = LowRankConv1x1(c1, c_, r=r)
        self.cv2 = LowRankConv1x1(c_ * 4, c2, r=r)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class BottleneckDS(nn.Module):
    """
    Bottleneck cải tiến cho C2f:
    - cv1 (1×1): LowRankConv1x1
    - cv2 (3×3): DSConv (depthwise + LR pointwise)
    - shortcut khi c1==c2 (giữ nguyên logic gốc)
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=1.0, r=None):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = LowRankConv1x1(c1, c_, r=r)
        self.cv2 = DSConv(c_, c2, k=3, s=1, r=r)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_LR(nn.Module):
    """
    C2f tối ưu theo bài báo:
    - cv1, cv2 (1×1): LowRankConv1x1
    - Bottleneck bên trong: BottleneckDS (DSC + LR)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, r=None, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = LowRankConv1x1(c1, 2 * self.c, r=r)
        self.cv2 = LowRankConv1x1((2 + n) * self.c, c2, r=r)
        self.m = nn.ModuleList(
            BottleneckDS(self.c, self.c, shortcut, g, e=1.0, r=r)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class GhostConv_LR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, r=32, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = LowRankConv1x1(c1, c_, r=r, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


# ============================================================
# ACBlock + DSConv_AC  (Asymmetric Convolution - ACNet ICCV2019)
# ============================================================
class ACBlock(nn.Module):
    """
    Asymmetric Conv Block: Training dùng 3 nhánh song song,
    Inference merge thành 1 conv 3×3 duy nhất (re-parameterization).
    Hỗ trợ depthwise (g=c1) → dùng trong DSConv_AC.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        assert k == 3, "ACBlock chỉ hỗ trợ kernel 3×3"
        self.deploy = deploy
        self.act = Conv.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        pad = autopad(k, p)

        if deploy:
            # Inference: 1 conv duy nhất
            self.conv = nn.Conv2d(c1, c2, k, s, pad, groups=g, bias=True)
        else:
            # Training: 3 nhánh, mỗi nhánh có BN riêng
            self.conv_3x3 = nn.Conv2d(c1, c2, (3,3), s, pad,   groups=g, bias=False)
            self.conv_1x3 = nn.Conv2d(c1, c2, (1,3), s, (0,1), groups=g, bias=False)
            self.conv_3x1 = nn.Conv2d(c1, c2, (3,1), s, (1,0), groups=g, bias=False)
            self.bn_3x3   = nn.BatchNorm2d(c2)
            self.bn_1x3   = nn.BatchNorm2d(c2)
            self.bn_3x1   = nn.BatchNorm2d(c2)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv(x))
        return self.act(
            self.bn_3x3(self.conv_3x3(x)) +
            self.bn_1x3(self.conv_1x3(x)) +
            self.bn_3x1(self.conv_3x1(x))
        )

    def _fuse_branch(self, conv_w, bn):
        """Fold BN vào conv weight → trả về (weight, bias) đã fuse."""
        std   = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_w = conv_w * scale
        fused_b = bn.bias - bn.running_mean * bn.weight / std
        return fused_w, fused_b

    def switch_to_deploy(self):
        """Merge 3 nhánh thành 1 conv 3×3, gọi trước khi inference/export."""
        if self.deploy:
            return
        # Fuse từng nhánh
        w3,   b3   = self._fuse_branch(self.conv_3x3.weight, self.bn_3x3)
        w1x3, b1x3 = self._fuse_branch(self.conv_1x3.weight, self.bn_1x3)
        w3x1, b3x1 = self._fuse_branch(self.conv_3x1.weight, self.bn_3x1)

        # Pad kernel bất đối xứng thành 3×3
        # w1x3: [C,c,1,3] → pad top/bottom → [C,c,3,3]
        # w3x1: [C,c,3,1] → pad left/right → [C,c,3,3]
        w1x3_pad = F.pad(w1x3, [0, 0, 1, 1])
        w3x1_pad = F.pad(w3x1, [1, 1, 0, 0])

        # Tổng hợp
        fused_w = w3 + w1x3_pad + w3x1_pad
        fused_b = b3 + b1x3 + b3x1

        # Tạo conv deploy
        self.conv = nn.Conv2d(
            self.conv_3x3.in_channels, self.conv_3x3.out_channels,
            3, self.conv_3x3.stride, self.conv_3x3.padding,
            groups=self.conv_3x3.groups, bias=True
        )
        self.conv.weight.data = fused_w
        self.conv.bias.data   = fused_b

        # Xoá nhánh training
        for attr in ['conv_3x3','conv_1x3','conv_3x1','bn_3x3','bn_1x3','bn_3x1']:
            delattr(self, attr)
        self.deploy = True


class DSConv_AC(nn.Module):
    """
    DSConv cải tiến: thay depthwise Conv 3×3 bằng ACBlock.
    Params inference = DSConv gốc.
    Accuracy tốt hơn nhờ training đa nhánh nắm bắt pattern ngang/dọc.
    Đặc biệt hữu ích với drone detection (vật thể góc nhìn trên xuống).
    """
    def __init__(self, c1, c2, k=3, s=1, r=None, act=True, deploy=False):
        super().__init__()
        self.dw = ACBlock(c1, c1, k, s, g=c1, act=False, deploy=deploy)  # depthwise AC
        self.pw = LowRankConv1x1(c1, c2, r=r, act=act)                   # pointwise LR (giữ nguyên)

    def forward(self, x):
        return self.pw(self.dw(x))

    def switch_to_deploy(self):
        self.dw.switch_to_deploy()


class MobileOneBlock(nn.Module):
    """
    MobileOneBlock: multi-branch re-parameterizable conv.

    Args:
        c1      : input channels
        c2      : output channels
        k       : number of parallel 3×3 branches (default 4)
        s       : stride
        g       : groups  (g=c1 → depthwise, dùng trong DSConv_MO)
        deploy  : True khi inference/export → dùng 1 conv duy nhất
    """

    def __init__(self, c1, c2, k=4, s=1, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.g = g
        self.c2 = c2
        self.act = Conv.default_act if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            # ── Inference mode: 1 conv 3×3 ──────────────────────────────
            self.reparam = nn.Conv2d(c1, c2, 3, s, 1, groups=g, bias=True)
        else:
            # ── Training: k nhánh 3×3, k nhánh 1×1, (optional) identity ─
            pad = autopad(3, None)

            # k branches 3×3
            self.branches_3x3 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c1, c2, 3, s, pad, groups=g, bias=False),
                    nn.BatchNorm2d(c2)
                ) for _ in range(k)
            ])

            # k branches 1×1  (luôn tạo, bất kể stride)
            self.branches_1x1 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False),
                    nn.BatchNorm2d(c2)
                ) for _ in range(k)
            ])

            # Identity branch (chỉ khi c1==c2 và stride=1)
            if c1 == c2 and s == 1:
                self.bn_identity = nn.BatchNorm2d(c2)
            else:
                self.bn_identity = None

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))

        out = sum(b(x) for b in self.branches_3x3)

        if self.branches_1x1 is not None:
            out = out + sum(b(x) for b in self.branches_1x1)

        if self.bn_identity is not None:
            out = out + self.bn_identity(x)

        return self.act(out)

    # ── Helper: fold BN vào conv ─────────────────────────────────────────
    @staticmethod
    def _fuse_conv_bn(conv, bn):
        w = conv.weight
        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_w = w * scale
        fused_b = bn.bias - bn.running_mean * bn.weight / std
        return fused_w, fused_b

    def switch_to_deploy(self):
        """Gọi trước khi inference hoặc export ONNX/TFLite."""
        if self.deploy:
            return

        # Tổng hợp weight từ tất cả nhánh 3×3
        K, B = 0.0, 0.0

        for seq in self.branches_3x3:
            w, b = self._fuse_conv_bn(seq[0], seq[1])
            K = K + w
            B = B + b

        # Pad 1×1 → 3×3 rồi cộng vào
        if self.branches_1x1 is not None:
            for seq in self.branches_1x1:
                w, b = self._fuse_conv_bn(seq[0], seq[1])
                # Pad 1×1 weight lên 3×3 (đặt vào tâm)
                # Stride đã được encode trong branches_3x3; 1×1 dùng cùng stride s,
                # nên chỉ cần pad spatial, không cần thêm xử lý stride.
                K = K + F.pad(w, [1, 1, 1, 1])
                B = B + b

        # Identity branch: BN(x)  =  BN(I * x)  →  pad identity kernel
        if self.bn_identity is not None:
            bn = self.bn_identity
            c = self.c2 // self.g
            # identity conv weight cho một group
            id_w = torch.zeros(self.c2, c, 3, 3,
                               device=bn.weight.device, dtype=bn.weight.dtype)
            for i in range(self.c2):
                id_w[i, i % c, 1, 1] = 1.0
            std = (bn.running_var + bn.eps).sqrt()
            scale = (bn.weight / std).reshape(-1, 1, 1, 1)
            K = K + id_w * scale
            B = B + bn.bias - bn.running_mean * bn.weight / std

        # Gán vào conv deploy
        self.reparam = nn.Conv2d(
            self.branches_3x3[0][0].in_channels,
            self.c2, 3,
            self.branches_3x3[0][0].stride,
            1, groups=self.g, bias=True
        )
        self.reparam.weight.data = K
        self.reparam.bias.data = B

        # Xóa nhánh training
        for attr in ['branches_3x3', 'branches_1x1', 'bn_identity']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        self.deploy = True


class DSConv_MO(nn.Module):
    """
    DSConv cải tiến: thay ACBlock bằng MobileOneBlock.
    DW (depthwise MobileOne) + PW (LowRankConv1x1 giữ nguyên)

    Dùng để thay thế DSConv_AC trong yaml:
        DSConv_AC  →  DSConv_MO
    """

    def __init__(self, c1, c2, k_branches=4, s=1, r=None, act=True, deploy=False):
        super().__init__()
        self.dw = MobileOneBlock(c1, c1, k=k_branches, s=s,
                                 g=c1, act=False, deploy=deploy)
        self.pw = LowRankConv1x1(c1, c2, r=r, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))

    def switch_to_deploy(self):
        self.dw.switch_to_deploy()


class PConv(nn.Module):
    """
    Partial Convolution (PConv) từ FasterNet (CVPR 2023).

    Chỉ apply conv 3×3 lên (c1 // n_div) channels đầu,
    phần còn lại copy qua không đổi.

    Args:
        c1      : input/output channels (PConv giữ channels)
        k       : kernel size (mặc định 3)
        n_div   : phân chia ratio — conv trên 1/n_div channels (mặc định 4)
        forward : 'split_cat' (nhanh hơn) hoặc 'slicing'
    """

    def __init__(self, c1, k=3, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv
        self.conv = nn.Conv2d(
            self.dim_conv, self.dim_conv,
            k, 1, k // 2, bias=False
        )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        else:
            self.forward = self.forward_split_cat

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class FasterNetBlock(nn.Module):
    """
    FasterNet Block: PConv + PWConv (pointwise) + activation.
    Giữ channels (c1 == c2), dùng như C2f bottleneck.

    Cấu trúc: PConv → BN+Act (middle) → PWConv → BN
    (Activation chỉ sau PConv để giảm memory)
    """

    def __init__(self, c1, n_div=4, act=True):
        super().__init__()
        self.pconv = PConv(c1, k=3, n_div=n_div)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = Conv.default_act if act is True else nn.Identity()
        self.pwconv = nn.Conv2d(c1, c1, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        shortcut = x
        x = self.act(self.bn1(self.pconv(x)))
        x = self.bn2(self.pwconv(x))
        return x + shortcut  # residual


class C2f_Faster(nn.Module):
    """
    C2f với FasterNet blocks bên trong — thay C2f_LR.

    Dùng thay thế C2f_LR trong yaml:
        C2f_LR  →  C2f_Faster
    """

    def __init__(self, c1, c2, n=1, shortcut=False, n_div=4, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            FasterNetBlock(self.c, n_div=n_div)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RELANBottleneck(nn.Module):
    """
    Bottleneck của R-ELAN: DW3×3 + PW1×1 với residual scaling.

    Args:
        c       : channels (input = output)
        scale   : residual scaling factor (YOLOv12 dùng 0.01 cho model lớn,
                  0.1 cho tiny — giúp train ổn định hơn)
    """

    def __init__(self, c, scale=0.1):
        super().__init__()
        self.scale = scale
        # DW 3×3 (spatial mixing) + BN + Act
        self.dw = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            Conv.default_act,
        )
        # PW 1×1 (channel mixing) + BN
        self.pw = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return x + self.scale * self.pw(self.dw(x))


class RELAN(nn.Module):
    """
    R-ELAN: Residual Efficient Layer Aggregation Network (YOLOv12, Feb 2025).
    Thay thế C2f_Faster trong head của YOLOv5-DP.

    Cấu trúc:
        x → cv1 (transition, điều chỉnh channels)
          → n × RELANBottleneck  (xử lý đệ quy với residual scaling)
          → concat tất cả intermediate features
          → cv2 (fusion 1×1)

    Args:
        c1          : input channels
        c2          : output channels
        n           : số bottleneck blocks (depth)
        shortcut    : không dùng (giữ tương thích với C2f interface)
        scale       : residual scale trong bottleneck (0.1 cho tiny model)
    """

    def __init__(self, c1, c2, n=1, shortcut=False, block_scale=0.1, scale=0.1):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, self.c, 1, 1)  # conv 1x1 (trong anh paper)
        self.cv2 = Conv((1 + n) * self.c, c2, 1, 1)  # fusion sau concatenation
        self.m = nn.ModuleList(
            RELANBottleneck(self.c, scale=scale) for _ in range(n)
        )
        self.block_scale = block_scale
        # self.use_block_res = (c1 == c2)

    def forward(self, x):
        y = [self.cv1(x)]
        for block in self.m:
            y.append(block(y[-1]))
        out = self.cv2(torch.cat(y, dim=1))
        # if self.use_block_res:
        #     out = out * self.block_scale + x
        return out


class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class UIBLite(nn.Module):
    """
    UIBLite - Universal Inverted Bottleneck Lite.

   Idea:
        x -> optional ExtraDW -> PW expand -> DW spatial -> PW project -> optional shortcut

    Dùng trong YAML:
        [-1, 1, UIBLite, [256]]
        [-1, 1, UIBLite, [256, 2.0]]
        [-1, 1, UIBLite, [256, 2.0, 3]]
        [-1, 1, UIBLite, [256, 2.0, 3, True]]
        [-1, 1, UIBLite, [256, 2.0, 3, True, False]]

    Args:
        c1          : input channels, parse_model tự truyền
        c2          : output channels
        e           : expansion ratio
        k           : depthwise kernel size
        shortcut    : dùng residual nếu c1 == c2
        extra_dw    : thêm depthwise trước expand, giống ExtraDW idea trong MobileNetV4 UIB
    """

    def __init__(self, c1, c2, e=2.0, k=3, shortcut=True, extra_dw=False):
        super().__init__()

        hidden = int(c2 * e)

        # Làm tròn channel về bội số 8 để thân thiện hơn với CPU/SIMD.
        hidden = max(8, int((hidden + 7) // 8 * 8))

        self.use_shortcut = shortcut and c1 == c2

        # Extra depthwise trước expand.
        # Mặc định tắt để graph đơn giản hơn cho Raspberry Pi.
        self.extra_dw = (
            Conv(c1, c1, k=3, s=1, g=c1)
            if extra_dw
            else nn.Identity()
        )

        # 1x1 pointwise expand
        self.expand = Conv(c1, hidden, k=1, s=1)

        # depthwise spatial mixing
        self.dw = Conv(hidden, hidden, k=k, s=1, g=hidden)

        # 1x1 pointwise project
        # act=False để giữ "linear bottleneck" ở output projection.
        self.project = Conv(hidden, c2, k=1, s=1, act=False)

    def forward(self, x):
        y = self.extra_dw(x)
        y = self.expand(y)
        y = self.dw(y)
        y = self.project(y)

        if self.use_shortcut:
            return x + y

        return y
