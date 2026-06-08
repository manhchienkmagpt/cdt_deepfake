import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

if "." in __package__:
    from ..evaluation.evaluator import evaluate
    from ..utils.checkpoint import load_checkpoint, save_checkpoint
    from .early_stopping import EarlyStopping
    from .losses import build_loss
    from .optimizers import build_optimizer
    from .schedulers import build_scheduler
else:
    from evaluation.evaluator import evaluate
    from training.early_stopping import EarlyStopping
    from training.losses import build_loss
    from training.optimizers import build_optimizer
    from training.schedulers import build_scheduler
    from utils.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, model: torch.nn.Module, config: Dict, device: torch.device, logger) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        self.criterion = build_loss(config).to(device)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        train_cfg = config.get("training", {})
        self.early_stopping = EarlyStopping(patience=int(train_cfg.get("early_stopping_patience", 5)))

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            if logits.ndim == 2 and logits.size(1) == 2:
                logits = logits[:, 1]
            return self.criterion(logits.float(), labels.float())
        return self.criterion(logits, labels.long())

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels, _ in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(images)
            logits = output["logits"] if isinstance(output, dict) else output
            loss = self._loss(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        self.logger.info("Epoch %03d | train_loss=%.6f", epoch, avg_loss)
        return avg_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[str, Dict[str, float]]:
        train_cfg = self.config.get("training", {})
        model_name = self.config.get("model_name", self.config.get("model", {}).get("name", "model"))
        save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = save_dir / f"{model_name}_best.pth"
        epochs = int(train_cfg.get("epochs", 50))
        best_metrics = {"AUC": -1.0}

        for epoch in range(1, epochs + 1):
            self.train_one_epoch(train_loader, epoch)
            val_metrics, _ = evaluate(self.model, val_loader, self.device)
            self.logger.info(
                "Epoch %03d | val Accuracy=%.4f F1_score=%.4f Precision=%.4f Recall=%.4f AUC=%.4f",
                epoch,
                val_metrics["Accuracy"],
                val_metrics["F1_score"],
                val_metrics["Precision"],
                val_metrics["Recall"],
                val_metrics["AUC"],
            )

            if val_metrics["AUC"] > best_metrics.get("AUC", -1.0):
                best_metrics = val_metrics
                save_checkpoint(str(best_path), self.model, self.optimizer, epoch, val_metrics)
                self.logger.info("Saved best checkpoint: %s", best_path)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.early_stopping.step(val_metrics["AUC"]):
                self.logger.info("Early stopping triggered at epoch %d", epoch)
                break

        load_checkpoint(str(best_path), self.model, map_location=self.device)
        return str(best_path), best_metrics

    def test_and_save(self, loader: DataLoader, dataset_name: str, results_dir: str) -> Dict[str, float]:
        metrics, _ = evaluate(self.model, loader, self.device)
        model_name = self.config.get("model_name", self.config.get("model", {}).get("name", "model"))
        output_path = Path(results_dir) / f"{model_name}_{dataset_name}_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(
            "%s | Accuracy=%.4f F1_score=%.4f Precision=%.4f Recall=%.4f AUC=%.4f",
            dataset_name,
            metrics["Accuracy"],
            metrics["F1_score"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["AUC"],
        )
        self.logger.info("Saved metrics: %s", output_path)
        return metrics
