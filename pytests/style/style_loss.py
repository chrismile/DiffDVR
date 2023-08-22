# Copyright (c) 2021-2023 Katherine Crowson, Christoph Neuhauser
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Adapted from: https://github.com/crowsonkb/style-transfer-pytorch

from enum import Enum
from functools import partial
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from . import sqrtm


class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:self.layers[-1] + 1]
        self.devices = [torch.device('cpu')] * len(self.model)

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices):
        for i, layer in enumerate(self.model):
            if i in devices:
                device = torch.device(devices[i])
            self.model[i] = layer.to(device)
            self.devices[i] = device

    def forward(self, input, layers=None):
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        input = self.normalize(input)
        for i in range(max(layers) + 1):
            input = self.model[i](input.to(self.devices[i]))
            if i in layers:
                feats[i] = input
        return feats


class ScaledMSELoss(nn.Module):
    """Computes MSE scaled such that its gradient L1 norm is approximately 1.
    This differs from Gatys at al. (2015) and Johnson et al."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def extra_repr(self):
        return f'eps={self.eps:g}'

    def forward(self, input, target):
        diff = input - target
        return diff.pow(2).sum() / diff.abs().sum().add(self.eps)


class PartialMSELoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)


class StyleLossGram(nn.Module):
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    @staticmethod
    def get_target(target):
        mat = target.flatten(-2)
        # The Gram matrix normalization differs from Gatys et al. (2015) and Johnson et al.
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input):
        return self.loss(self.get_target(input), self.target)


def eye_like(x):
    return torch.eye(x.shape[-2], x.shape[-1], dtype=x.dtype, device=x.device).expand_as(x)


class StyleLossW2(nn.Module):
    """Wasserstein-2 style loss."""

    def __init__(self, target, eps=1e-4):
        super().__init__()
        self.sqrtm = partial(sqrtm.sqrtm_ns_lyap, num_iters=12)
        mean, srm = target
        cov = self.srm_to_cov(mean, srm) + eye_like(srm) * eps
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)
        self.register_buffer('cov_sqrt', self.sqrtm(cov))
        self.register_buffer('eps', mean.new_tensor(eps))

    @staticmethod
    def get_target(target):
        """Compute the mean and second raw moment of the target activations.
        Unlike the covariance matrix, these are valid to combine linearly."""
        mean = target.mean([-2, -1])
        srm = torch.einsum('...chw,...dhw->...cd', target, target) / (target.shape[-2] * target.shape[-1])
        return mean, srm

    @staticmethod
    def srm_to_cov(mean, srm):
        """Compute the covariance matrix from the mean and second raw moment."""
        return srm - torch.einsum('...c,...d->...cd', mean, mean)

    def forward(self, input):
        mean, srm = self.get_target(input)
        cov = self.srm_to_cov(mean, srm) + eye_like(srm) * self.eps
        mean_diff = torch.mean((mean - self.mean) ** 2)
        sqrt_term = self.sqrtm(self.cov_sqrt @ cov @ self.cov_sqrt)
        cov_diff = torch.diagonal(self.cov + cov - 2 * sqrt_term, dim1=-2, dim2=-1).mean()
        return mean_diff + cov_diff


class SumLoss(nn.ModuleList):
    def __init__(self, losses, verbose=False):
        super().__init__(losses)
        self.verbose = verbose

    def forward(self, *args, **kwargs):
        losses = [loss(*args, **kwargs) for loss in self]
        if self.verbose:
            for i, loss in enumerate(losses):
                print(f'({i}): {loss.item():g}')
        return sum(loss.to(losses[-1].device) for loss in losses)


class Scale(nn.Module):
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'(scale): {self.scale.item():g}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale


class LayerApply(nn.Module):
    def __init__(self, module, layer):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self):
        return f'(layer): {self.layer!r}'

    def forward(self, input):
        return self.module(input[self.layer])


class StyleLossType(Enum):
    W2 = 0     # Wasserstein-2 loss
    GRAM = 1   # Gram matrix-based loss
    L2 = 2     # L2 loss (MSE)


class StyleLoss(nn.Module):
    def __init__(self, style_image, loss_type=StyleLossType.W2, pooling='max'):
        super().__init__()
        self.loss_type = loss_type
        self.style_layers = [1, 6, 11, 20, 29]
        style_weights = [256, 64, 16, 4, 1]
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]
        self.model = VGGFeatures(self.style_layers, pooling=pooling)

        style_losses = []
        if loss_type == StyleLossType.L2:
            self.loss_sum = PartialMSELoss(style_image)
        else:
            style_feats = self.model(style_image, layers=self.style_layers)
            for layer, weight in zip(self.style_layers, self.style_weights):
                if loss_type == StyleLossType.W2:
                    target_mean, target_cov = StyleLossW2.get_target(style_feats[layer])
                    style_loss = StyleLossW2((target_mean, target_cov))
                elif loss_type == StyleLossType.GRAM:
                    gram_mat = StyleLossGram.get_target(style_feats[layer])
                    style_loss = StyleLossGram(gram_mat)
                else:
                    raise Exception('Error in StyleLoss.__init__: Invalid loss type.')
                style_losses.append(Scale(LayerApply(style_loss, layer), weight))
            self.loss_sum = SumLoss(style_losses)

    def forward(self, input):
        if self.loss_type == StyleLossType.L2:
            feats = input
        else:
            feats = self.model(input)
        loss = self.loss_sum(feats)
        return loss
