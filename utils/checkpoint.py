from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        ckpt_path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, map_location: Any = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

