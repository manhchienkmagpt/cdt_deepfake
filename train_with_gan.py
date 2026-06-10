import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if __package__:
    from .datasets import DeepfakeImageFolderDataset, build_transforms
    from .datasets.image_folder_dataset import SUPPORTED_EXTENSIONS
    from .models import build_model
    from .training import Trainer
    from .utils import load_config, set_seed
    from .utils.logger import setup_logger
else:
    sys.path.insert(0, str(ROOT))
    from datasets import DeepfakeImageFolderDataset, build_transforms
    from datasets.image_folder_dataset import SUPPORTED_EXTENSIONS
    from models import build_model
    from training import Trainer
    from utils import load_config, set_seed
    from utils.logger import setup_logger


class ExtraFakeFolderDataset(Dataset):
    """Read extra fake images from a folder and label every image as fake."""

    def __init__(
        self,
        root: str,
        transform=None,
        extensions: Optional[Iterable[str]] = None,
        recursive: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.extensions = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
        self.recursive = recursive

        if not self.root.exists():
            raise FileNotFoundError(f"GAN fake folder does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"GAN fake root must be a folder: {self.root}")

        self.samples = self._scan_samples()
        if not self.samples:
            raise RuntimeError(f"No supported GAN fake images found in: {self.root}")

    def _scan_samples(self) -> List[Path]:
        paths = self.root.rglob("*") if self.recursive else self.root.iterdir()
        return sorted(
            path for path in paths if path.is_file() and path.suffix.lower() in self.extensions
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 1, str(image_path)


def build_loader(config, dataset_name: str, split: str, train: bool, gan_fake_root: Optional[str] = None):
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
    extra_fake_count = 0
    if train and dataset_name == "ffpp" and gan_fake_root:
        gan_dataset = ExtraFakeFolderDataset(
            gan_fake_root,
            transform=transform,
            recursive=bool(data_cfg.get("gan_fake_recursive", True)),
        )
        extra_fake_count = len(gan_dataset)
        dataset = ConcatDataset([dataset, gan_dataset])
        counts = {
            "real": counts["real"],
            "fake": counts["fake"] + extra_fake_count,
            "total": counts["total"] + extra_fake_count,
        }

    upsample_text = f" real_upsample_factor={real_upsample_factor}" if train else ""
    gan_text = f" gan_fake={extra_fake_count}" if extra_fake_count else ""
    print(
        f"{dataset_name.upper()} {split}: real={counts['real']} "
        f"fake={counts['fake']} total={counts['total']}{upsample_text}{gan_text}"
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=train,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train deepfake detector with extra GAN fake images added to FF++ train split."
    )
    parser.add_argument("--model", required=True, choices=["efficientb4", "fwa", "ucf", "srm", "spsl"])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gan-fake-root",
        default=None,
        help="Folder containing extra GAN fake images to add only during training.",
    )
    parser.add_argument(
        "--real-upsample-factor",
        type=int,
        default=None,
        help="Override data.real_upsample_factor for train split. 1 disables real upsampling.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_name"] = args.model
    data_cfg = config.setdefault("data", {})
    if args.real_upsample_factor is not None:
        data_cfg["real_upsample_factor"] = args.real_upsample_factor

    gan_fake_root = args.gan_fake_root or data_cfg.get("gan_fake_root")
    if not gan_fake_root:
        raise ValueError("Provide --gan-fake-root or set data.gan_fake_root in the config.")

    set_seed(int(config.get("training", {}).get("seed", 1024)))
    logger = setup_logger(config.get("logging", {}).get("log_dir", "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("cuda", True) else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Adding GAN fake images from: %s", gan_fake_root)

    train_loader = build_loader(config, "ffpp", "train", train=True, gan_fake_root=gan_fake_root)
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
