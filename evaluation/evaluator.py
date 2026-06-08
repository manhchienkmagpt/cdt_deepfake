from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], List[str]]:
    model.eval()
    labels = []
    probs = []
    paths = []

    for images, batch_labels, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True).long()
        output = model(images)
        logits = output["logits"] if isinstance(output, dict) else output
        batch_probs = torch.softmax(logits, dim=1)[:, 1]

        labels.extend(batch_labels.detach().cpu().tolist())
        probs.extend(batch_probs.detach().cpu().tolist())
        paths.extend(list(batch_paths))

    return compute_metrics(labels, probs), paths

