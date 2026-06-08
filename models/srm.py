import math
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import Xception, load_xception_pretrained


class SRMConv2dSimple(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super().__init__()
        self.trunc = nn.Hardtanh(-3, 3)
        self.kernel = nn.Parameter(self._build_kernel(inc), requires_grad=learnable)

    def forward(self, x):
        return self.trunc(F.conv2d(x, self.kernel, stride=1, padding=2))

    def _build_kernel(self, inc):
        filter1 = np.asarray([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=float) / 4.0
        filter2 = np.asarray([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=float) / 12.0
        filter3 = np.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float) / 2.0
        filters = np.array([[filter1], [filter2], [filter3]])
        filters = np.repeat(filters, inc, axis=1)
        return torch.FloatTensor(filters)


class SRMConv2dSeparate(nn.Module):
    def __init__(self, inc, outc, learnable=False):
        super().__init__()
        self.inc = inc
        self.trunc = nn.Hardtanh(-3, 3)
        self.kernel = nn.Parameter(self._build_kernel(inc), requires_grad=learnable)
        self.out_conv = nn.Sequential(nn.Conv2d(3 * inc, outc, 1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        return self.out_conv(self.trunc(out))

    def _build_kernel(self, inc):
        filter1 = np.asarray([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=float) / 4.0
        filter2 = np.asarray([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=float) / 12.0
        filter3 = np.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float) / 2.0
        filters = np.array([[filter1], [filter2], [filter3]])
        filters = np.repeat(filters, inc, axis=0)
        return torch.FloatTensor(filters)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.shared(self.avg_pool(x)) + self.shared(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avgout, maxout], dim=1)))


class DualCrossModalAttention(nn.Module):
    def __init__(self, in_dim, size=16, ratio=8):
        super().__init__()
        self.key_conv1 = nn.Conv2d(in_dim, in_dim // ratio, 1)
        self.key_conv2 = nn.Conv2d(in_dim, in_dim // ratio, 1)
        self.key_conv_share = nn.Conv2d(in_dim // ratio, in_dim // ratio, 1)
        self.value_conv1 = nn.Conv2d(in_dim, in_dim, 1)
        self.value_conv2 = nn.Conv2d(in_dim, in_dim, 1)
        self.linear1 = nn.Linear(size * size, size * size)
        self.linear2 = nn.Linear(size * size, size * size)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        b, c, h, w = x.size()
        key1 = self.key_conv_share(self.key_conv1(x)).view(b, -1, h * w).permute(0, 2, 1)
        key2 = self.key_conv_share(self.key_conv2(y)).view(b, -1, h * w)
        energy = torch.bmm(key1, key2)
        att_y_on_x = self.softmax(self.linear1(energy))
        att_x_on_y = self.softmax(self.linear2(energy.permute(0, 2, 1)))

        value_y = self.value_conv2(y).view(b, -1, h * w)
        out_x = torch.bmm(value_y, att_y_on_x.permute(0, 2, 1)).view(b, c, h, w)
        value_x = self.value_conv1(x).view(b, -1, h * w)
        out_y = torch.bmm(value_x, att_x_on_y.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma1 * out_x + x, self.gamma2 * out_y + y


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.srm = SRMConv2dSimple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pa = SpatialAttention()

    def forward(self, x):
        return self.pa(self.conv(self.srm(x)))


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=4096, out_chan=2048):
        super().__init__()
        self.convblk = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1, bias=False), nn.BatchNorm2d(out_chan), nn.ReLU())
        self.ca = ChannelAttention(out_chan, ratio=16)

    def forward(self, x, y):
        fuse = self.convblk(torch.cat((x, y), dim=1))
        return fuse * self.ca(fuse)


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma=0.8, dim=2):
        super().__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij")
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size()).repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer("weight", kernel)
        self.groups = channels

    def forward(self, x):
        if self.training:
            return F.conv2d(x, weight=self.weight, groups=self.groups, padding=self.kernel_size // 2)
        return x


class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1, clip=1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.clip = clip

    def forward(self, x):
        if self.training:
            noise = x.new_empty(x.size()).normal_(self.mean, self.std)
            return torch.clamp(x + noise, -self.clip, self.clip)
        return x


class SRMDetector(nn.Module):
    """SRM high-frequency two-branch Xception detector."""

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        backbone_cfg = model_cfg.get("backbone_config", {"num_classes": 2, "inc": 3, "mode": "original"})
        self.backbone_rgb = Xception(backbone_cfg)
        self.backbone_srm = Xception(backbone_cfg)
        pretrained = model_cfg.get("pretrained", "")
        load_xception_pretrained(self.backbone_rgb, pretrained)
        load_xception_pretrained(self.backbone_srm, pretrained)

        self.noise = GaussianNoise(clip=1)
        self.blur = GaussianSmoothing(channels=3, kernel_size=7, sigma=0.8)
        self.srm_conv0 = SRMConv2dSimple(inc=3)
        self.srm_conv1 = SRMConv2dSeparate(32, 32)
        self.srm_conv2 = SRMConv2dSeparate(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dual_cma0 = DualCrossModalAttention(in_dim=728)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728)
        self.fusion = FeatureFusionModule()

    def features(self, images):
        x = self.noise(images)
        x = self.blur(x)
        srm = self.srm_conv0(x)

        x = self.backbone_rgb.fea_part1_0(x)
        y = self.relu(self.backbone_srm.fea_part1_0(srm) + self.srm_conv1(x))
        x = self.backbone_rgb.fea_part1_1(x)
        y = self.relu(self.backbone_srm.fea_part1_1(y) + self.srm_conv2(x))

        att_map = self.srm_sa(srm)
        x = self.srm_sa_post(x * att_map + x)
        x = self.backbone_rgb.fea_part2(x)
        y = self.backbone_srm.fea_part2(y)
        x, y = self.dual_cma0(x, y)
        x = self.backbone_rgb.fea_part3(x)
        y = self.backbone_srm.fea_part3(y)
        x, y = self.dual_cma1(x, y)
        x = self.backbone_rgb.fea_part5(self.backbone_rgb.fea_part4(x))
        y = self.backbone_srm.fea_part5(self.backbone_srm.fea_part4(y))
        return self.fusion(x, y)

    def forward(self, images: torch.Tensor):
        features = self.features(images)
        logits = self.backbone_rgb.classifier(features)
        return {"logits": logits, "prob": torch.softmax(logits, dim=1)[:, 1], "features": features}
