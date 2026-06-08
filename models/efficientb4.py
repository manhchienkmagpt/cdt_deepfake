import torch
import torch.nn as nn

from .backbones import EfficientNetB4Backbone


class EfficientB4Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        backbone_cfg = dict(model_cfg.get("backbone_config", {}))
        backbone_cfg["pretrained"] = model_cfg.get("pretrained", None)
        self.backbone = EfficientNetB4Backbone(backbone_cfg)

    def forward(self, images: torch.Tensor):
        features = self.backbone.features(images)
        logits = self.backbone.classifier(features)
        return {"logits": logits, "prob": torch.softmax(logits, dim=1)[:, 1], "features": features}

