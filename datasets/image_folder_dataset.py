from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FFPP_CLASS_TO_LABEL = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceShifter": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
}

CELEBDF_CLASS_TO_LABEL = {
    "real": 0,
    "fake": 1,
}


class DeepfakeImageFolderDataset(Dataset):
    """Read direct image files from FF++ or CelebDF class folders.

    Each item returns (image_tensor, binary_label, image_path). Labels follow:
    real/original = 0 and fake/manipulated = 1.
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        split: Optional[str] = None,
        transform=None,
        extensions: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset.lower()
        self.split = split
        self.transform = transform
        self.extensions = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}

        if self.dataset == "ffpp":
            if split not in {"train", "val", "test"}:
                raise ValueError("FF++ split must be one of: train, val, test")
            self.scan_root = self.root / split
            self.class_to_label = FFPP_CLASS_TO_LABEL
        elif self.dataset == "celebdf":
            self.scan_root = self.root
            self.class_to_label = CELEBDF_CLASS_TO_LABEL
        else:
            raise ValueError("dataset must be 'ffpp' or 'celebdf'")

        if not self.scan_root.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {self.scan_root}")

        self.samples = self._scan_samples()
        if not self.samples:
            raise RuntimeError(f"No supported images found in: {self.scan_root}")

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        missing_classes = []

        for class_name, label in self.class_to_label.items():
            class_dir = self.scan_root / class_name
            if not class_dir.exists():
                missing_classes.append(str(class_dir))
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in self.extensions:
                    samples.append((image_path, label))

        if missing_classes:
            print("Warning: missing class folders:")
            for folder in missing_classes:
                print(f"  - {folder}")

        return samples

    def class_counts(self) -> Dict[str, int]:
        real = sum(1 for _, label in self.samples if label == 0)
        fake = sum(1 for _, label in self.samples if label == 1)
        return {"real": real, "fake": fake, "total": len(self.samples)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(image_path)

