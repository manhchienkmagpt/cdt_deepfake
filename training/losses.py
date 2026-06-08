import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    """AM-Softmax loss used by DeepfakeBench SRM."""

    def __init__(self, margin_type: str = "cos", gamma: float = 0.0, m: float = 0.45, s: float = 30.0, t: float = 1.0):
        super().__init__()
        if margin_type not in {"cos", "arc"}:
            raise ValueError("margin_type must be 'cos' or 'arc'")
        self.margin_type = margin_type
        self.gamma = gamma
        self.m = m
        self.s = s
        self.t = t
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.margin_type == "cos":
            phi_theta = logits - self.m
        else:
            sine = torch.sqrt(torch.clamp(1.0 - torch.pow(logits, 2), min=0.0))
            phi_theta = logits * self.cos_m - sine * self.sin_m
            phi_theta = torch.where(logits > self.th, phi_theta, logits - self.sin_m * self.m)

        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        output = torch.where(index, phi_theta, logits)
        losses = F.cross_entropy(self.s * output, targets, reduction="none")
        if self.gamma > 0:
            p = torch.exp(-losses)
            losses = (1 - p) ** self.gamma * losses
        return losses.mean()


def build_loss(config: Dict, class_weight: Optional[torch.Tensor] = None) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy").lower()
    if class_weight is None and loss_cfg.get("class_weight") is not None:
        class_weight = torch.tensor(loss_cfg["class_weight"], dtype=torch.float32)

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weight)
    if loss_type == "am_softmax":
        params = loss_cfg.get("params", {"gamma": 0.0, "m": 0.45, "s": 30.0, "t": 1.0})
        return AMSoftmaxLoss(**params)
    if loss_type == "bce_with_logits":
        pos_weight = loss_cfg.get("pos_weight", None)
        tensor_weight = None if pos_weight is None else torch.tensor([float(pos_weight)])
        return nn.BCEWithLogitsLoss(pos_weight=tensor_weight)
    raise ValueError(f"Unsupported loss type: {loss_type}")
