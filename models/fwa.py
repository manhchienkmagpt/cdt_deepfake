import torch
import torch.nn as nn

from .backbones import Xception, load_xception_pretrained


class FWADetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        self.backbone = Xception(model_cfg.get("backbone_config", {}))
        load_xception_pretrained(self.backbone, model_cfg.get("pretrained", ""))

    def forward(self, images: torch.Tensor):
        features = self.backbone.features(images)
        logits = self.backbone.classifier(features)
        return {"logits": logits, "prob": torch.softmax(logits, dim=1)[:, 1], "features": features}

