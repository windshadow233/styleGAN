import torch
from torch import nn
from torch.nn import functional as F
import math


class EqualizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 gain=2 ** 0.5, stride=1, padding=1, bias=True):
        super(EqualizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight.data.normal_(0.0, 1.0)
        if self.bias is not None:
            self.bias.data.fill_(0)
        fan_in = kernel_size * kernel_size * in_channels
        self.scale = gain * math.sqrt(1. / fan_in)

    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.weight.mul(self.scale),  # scale the weight on runtime
                        bias=self.bias,
                        stride=self.stride, padding=self.padding)


class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, gain=2 ** 0.5, bias=True):
        super(EqualizedLinear, self).__init__(in_features, out_features, bias)
        self.weight.data.normal_(0.0, 1.0)
        if self.bias is not None:
            self.bias.data.fill_(0)
        fan_in = in_features
        self.scale = gain * math.sqrt(1. / fan_in)

    def forward(self, x):
        return F.linear(x, weight=self.weight.mul(self.scale), bias=self.bias)


class PixelNormLayer(nn.Module):
    """
    论文没用到
    """
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        channels = x.shape[1]
        return x / x.norm(dim=1, keepdim=True) * math.sqrt(channels)


class AdaIN(nn.Module):
    def __init__(self, eps=1e-8):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        mu = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.var(x, dim=(2, 3), keepdim=True) + self.eps
        ys, yb = torch.chunk(y[..., None, None], 2, dim=1)
        return ys * (x - mu) / var.sqrt() + yb


class AddNoise(nn.Module):
    def __init__(self, channels):
        super(AddNoise, self).__init__()
        self.scaling_factor = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        added_noise = torch.randn_like(x, device=x.device)
        return x + added_noise * self.scaling_factor.view(1, -1, 1, 1)


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return f'shape={self.shape}'


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class GConvLayer(nn.Module):
    """Conv + Noise + AdaIN"""
    def __init__(self, in_channels, out_channels, kernel_size=3, use_conv=True,
                 stride=1, padding=1, equalize_lr=True, use_leaky=True, negative_slope=0.2, w_dim=512):
        super(GConvLayer, self).__init__()
        self.use_conv = use_conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_fcn = nn.LeakyReLU(negative_slope) if use_leaky else nn.ReLU()
        if equalize_lr:
            if use_conv:
                self.conv_layer = EqualizedConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.affine = EqualizedLinear(w_dim, 2 * out_channels, gain=1)
        else:
            if use_conv:
                self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.affine = nn.Linear(w_dim, 2 * out_channels)
        self.affine.bias.data = torch.cat([torch.ones(out_channels), torch.zeros(out_channels)])
        self.add_noise = AddNoise(out_channels)
        self.adaIN = AdaIN()

    def forward(self, x, w):
        if self.use_conv:
            x = self.conv_layer(x)
        x = self.add_noise(x)
        y = self.affine(w)
        y = self.act_fcn(y)
        x = self.adaIN(x, y)
        x = self.act_fcn(x)
        return x


class GConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, input_constant=False,
                 w_dim=512, use_leaky=True, negative_slope=0.2, equalize_lr=True):
        super(GConvBlock, self).__init__()
        self.layers = nn.ModuleList([
            GConvLayer(in_channels, out_channels, use_conv=not input_constant,
                       use_leaky=use_leaky, negative_slope=negative_slope,
                       w_dim=w_dim, equalize_lr=equalize_lr),
            GConvLayer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                       use_leaky=use_leaky, negative_slope=negative_slope,
                       w_dim=w_dim, equalize_lr=equalize_lr)
        ])

    def forward(self, x, w):
        x = self.layers[0](x, w)
        x = self.layers[1](x, w)
        return x


class DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_leaky=True, negative_slope=0.2, equalize_lr=True):
        super(DConvBlock, self).__init__()
        act_fcn = nn.LeakyReLU(negative_slope) if use_leaky else nn.ReLU()
        if equalize_lr:
            conv_layer = EqualizedConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        else:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding), act_fcn
        self.layers = nn.Sequential(conv_layer, act_fcn)

    def forward(self, x):
        return self.layers(x)


class MinibatchStatConcatLayer(nn.Module):
    def __init__(self):
        super(MinibatchStatConcatLayer, self).__init__()
        self.adjust_std = torch.std

    def forward(self, x):
        shape = x.shape
        batch_std = self.adjust_std(x, dim=0, keepdim=True)
        vals = torch.mean(batch_std)
        vals = vals.repeat(shape[0], 1, shape[2], shape[3])
        return torch.cat([x, vals], dim=1)
