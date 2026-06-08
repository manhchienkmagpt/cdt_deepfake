import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from chien_deepfake.datasets import DeepfakeImageFolderDataset, build_transforms
from chien_deepfake.models import build_model
from chien_deepfake.training import Trainer
from chien_deepfake.utils import load_config, set_seed
from chien_deepfake.utils.logger import setup_logger


def build_loader(config, dataset_name: str, split: str, train: bool):
    data_cfg = config.get("data", {})
    transform = build_transforms(config, train=train)
    if dataset_name == "ffpp":
        dataset = DeepfakeImageFolderDataset(data_cfg["ffpp_root"], dataset="ffpp", split=split, transform=transform)
    elif dataset_name == "celebdf":
        dataset = DeepfakeImageFolderDataset(data_cfg["celebdf_test_root"], dataset="celebdf", transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    counts = dataset.class_counts()
    print(f"{dataset_name.upper()} {split}: real={counts['real']} fake={counts['fake']} total={counts['total']}")
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=train,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector on flat FF++ image folders.")
    parser.add_argument("--model", required=True, choices=["efficientb4", "fwa", "ucf", "srm", "spsl"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_name"] = args.model
    set_seed(int(config.get("training", {}).get("seed", 1024)))
    logger = setup_logger(config.get("logging", {}).get("log_dir", "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("cuda", True) else "cpu")
    logger.info("Using device: %s", device)

    train_loader = build_loader(config, "ffpp", "train", train=True)
    val_loader = build_loader(config, "ffpp", "val", train=False)
    ffpp_test_loader = build_loader(config, "ffpp", "test", train=False)
    celebdf_test_loader = build_loader(config, "celebdf", "test", train=False)

    model = build_model(args.model, config)
    trainer = Trainer(model, config, device, logger)
    best_path, best_metrics = trainer.fit(train_loader, val_loader)
    logger.info("Best checkpoint: %s | val AUC=%.4f", best_path, best_metrics["AUC"])

    results_dir = config.get("evaluation", {}).get("results_dir", "results")
    trainer.test_and_save(ffpp_test_loader, "ffpp_test", results_dir)
    trainer.test_and_save(celebdf_test_loader, "celebdf_test", results_dir)


if __name__ == "__main__":
    main()

