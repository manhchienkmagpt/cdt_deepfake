from typing import Dict, Optional

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = config.get("scheduler", None)
    if not sched_cfg:
        return None

    sched_type = sched_cfg.get("type", "none").lower()
    if sched_type in {"none", "null"}:
        return None
    if sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("t_max", config.get("training", {}).get("epochs", 50))),
        )
    raise ValueError(f"Unsupported scheduler type: {sched_type}")

