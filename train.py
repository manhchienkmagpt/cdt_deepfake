import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if __package__:
    from .datasets import DeepfakeImageFolderDataset, build_transforms
    from .models import build_model
    from .training import Trainer
    from .utils import load_config, set_seed
    from .utils.logger import setup_logger
else:
    sys.path.insert(0, str(ROOT))
    from datasets import DeepfakeImageFolderDataset, build_transforms
    from models import build_model
    from training import Trainer
    from utils import load_config, set_seed
    from utils.logger import setup_logger


def build_loader(config, dataset_name: str, split: str, train: bool):
    data_cfg = config.get("data", {})
    transform = build_transforms(config, train=train)
    real_upsample_factor = int(data_cfg.get("real_upsample_factor", 1)) if train else 1
    if dataset_name == "ffpp":
        dataset = DeepfakeImageFolderDataset(
            data_cfg["ffpp_root"],
            dataset="ffpp",
            split=split,
            transform=transform,
            real_upsample_factor=real_upsample_factor,
        )
    elif dataset_name == "celebdf":
        dataset = DeepfakeImageFolderDataset(
            data_cfg["celebdf_test_root"],
            dataset="celebdf",
            transform=transform,
            real_upsample_factor=real_upsample_factor,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    counts = dataset.class_counts()
    upsample_text = f" real_upsample_factor={real_upsample_factor}" if train else ""
    print(
        f"{dataset_name.upper()} {split}: real={counts['real']} "
        f"fake={counts['fake']} total={counts['total']}{upsample_text}"
    )
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
    parser.add_argument(
        "--real-upsample-factor",
        type=int,
        default=None,
        help="Override data.real_upsample_factor for train split. 1 disables real upsampling.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_name"] = args.model
    if args.real_upsample_factor is not None:
        config.setdefault("data", {})["real_upsample_factor"] = args.real_upsample_factor
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
