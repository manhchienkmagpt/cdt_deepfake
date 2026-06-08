from typing import Dict

import torch


def build_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    optim_cfg = config.get("optimizer", {})
    opt_type = optim_cfg.get("type", "adam").lower()
    lr = float(optim_cfg.get("learning_rate", config.get("training", {}).get("learning_rate", 2e-4)))
    weight_decay = float(optim_cfg.get("weight_decay", config.get("training", {}).get("weight_decay", 5e-4)))

    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
            eps=float(optim_cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
            amsgrad=bool(optim_cfg.get("amsgrad", False)),
        )
    if opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer type: {opt_type}")

