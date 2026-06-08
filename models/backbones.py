import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = None
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)

        relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.extend([relu, SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False), nn.BatchNorm2d(out_filters)])
            filters = out_filters
        for _ in range(reps - 1):
            rep.extend([relu, SeparableConv2d(filters, filters, 3, 1, 1, bias=False), nn.BatchNorm2d(filters)])
        if not grow_first:
            rep.extend([relu, SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False), nn.BatchNorm2d(out_filters)])
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        skip = self.skipbn(self.skip(inp)) if self.skip is not None else inp
        return x + skip


class Xception(nn.Module):
    """Xception backbone adapted from DeepfakeBench."""

    def __init__(self, config: Dict):
        super().__init__()
        self.num_classes = int(config.get("num_classes", 2))
        self.mode = config.get("mode", "original")
        inc = int(config.get("inc", 3))
        dropout = config.get("dropout", False)

        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        self.block4 = Block(728, 728, 3, 1)
        self.block5 = Block(728, 728, 3, 1)
        self.block6 = Block(728, 728, 3, 1)
        self.block7 = Block(728, 728, 3, 1)
        self.block8 = Block(728, 728, 3, 1)
        self.block9 = Block(728, 728, 3, 1)
        self.block10 = Block(728, 728, 3, 1)
        self.block11 = Block(728, 728, 3, 1)
        self.block12 = Block(728, 1024, 2, 2, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        final_channel = 512 if self.mode == "adjust_channel" else 2048
        self.last_linear = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(final_channel, self.num_classes)) if dropout else nn.Linear(final_channel, self.num_classes)
        self.adjust_channel = nn.Sequential(nn.Conv2d(2048, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=False))

    def fea_part1_0(self, x):
        return self.relu(self.bn1(self.conv1(x)))

    def fea_part1_1(self, x):
        return self.relu(self.bn2(self.conv2(x)))

    def fea_part1(self, x):
        return self.fea_part1_1(self.fea_part1_0(x))

    def fea_part2(self, x):
        return self.block3(self.block2(self.block1(x)))

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        return self.block7(self.block6(self.block5(self.block4(x))))

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            return self.block12(x)
        return self.block12(self.block11(self.block10(self.block9(self.block8(x)))))

    def fea_part5(self, x):
        x = self.relu(self.bn3(self.conv3(x)))
        return self.bn4(self.conv4(x))

    def features(self, x):
        x = self.fea_part5(self.fea_part4(self.fea_part3(self.fea_part2(self.fea_part1(x)))))
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, features):
        x = features if self.mode == "adjust_channel" else self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        self.last_emb = x
        return self.last_linear(x)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class EfficientNetB4Backbone(nn.Module):
    """EfficientNet-B4 wrapper matching DeepfakeBench's feature/classifier split."""

    def __init__(self, config: Dict):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as exc:
            raise ImportError("Install efficientnet_pytorch to use EfficientB4: pip install efficientnet_pytorch") from exc

        self.num_classes = int(config.get("num_classes", 2))
        self.dropout = config.get("dropout", False)
        self.mode = config.get("mode", "original")
        inc = int(config.get("inc", 3))
        pretrained = config.get("pretrained", None)
        if pretrained and Path(pretrained).exists():
            self.efficientnet = EfficientNet.from_pretrained("efficientnet-b4", weights_path=pretrained)
        else:
            self.efficientnet = EfficientNet.from_name("efficientnet-b4")

        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        self.dropout_layer = nn.Dropout(p=float(self.dropout)) if self.dropout else None
        self.last_layer = nn.Linear(1792, self.num_classes)
        self.adjust_channel = nn.Sequential(nn.Conv2d(1792, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def features(self, x):
        x = self.efficientnet.extract_features(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, features):
        x = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        self.last_emb = x
        return self.last_layer(x)

    def forward(self, x):
        return self.classifier(self.features(x))


def load_xception_pretrained(model: nn.Module, pretrained_path: str, strict: bool = False) -> None:
    if not pretrained_path:
        return
    path = Path(pretrained_path)
    if not path.exists():
        logger.warning("Pretrained file not found, training from scratch: %s", path)
        return

    state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model_state = model.state_dict()
    cleaned = {}
    for name, weights in state_dict.items():
        name = name.replace("module.", "")
        if "fc" in name or "last_linear" in name:
            continue
        if "pointwise" in name and weights.ndim == 2:
            weights = weights.unsqueeze(-1).unsqueeze(-1)
        if name in model_state and model_state[name].shape != weights.shape:
            logger.warning("Skip pretrained layer with mismatched shape: %s", name)
            continue
        cleaned[name] = weights
    model.load_state_dict(cleaned, strict=strict)
    logger.info("Loaded Xception pretrained weights from %s", path)
