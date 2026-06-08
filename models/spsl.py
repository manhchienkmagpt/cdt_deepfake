import torch
import torch.nn as nn

from .backbones import Xception, load_xception_pretrained


class SPSLDetector(nn.Module):
    """SPSL keeps DeepfakeBench's spatial phase channel before Xception."""

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        self.backbone = Xception(model_cfg.get("backbone_config", {"inc": 4}))
        pretrained = model_cfg.get("pretrained", "")
        if pretrained:
            load_xception_pretrained(self.backbone, pretrained)

    def phase_without_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        gray = torch.mean(images, dim=1, keepdim=True)
        spectrum = torch.fft.fftn(gray, dim=(-1, -2))
        phase = torch.angle(spectrum)
        reconstructed = torch.exp(1j * phase)
        return torch.real(torch.fft.ifftn(reconstructed, dim=(-1, -2)))

    def forward(self, images: torch.Tensor):
        phase = self.phase_without_amplitude(images)
        x = torch.cat((images, phase), dim=1)
        features = self.backbone.features(x)
        logits = self.backbone.classifier(features)
        return {"logits": logits, "prob": torch.softmax(logits, dim=1)[:, 1], "features": features}

