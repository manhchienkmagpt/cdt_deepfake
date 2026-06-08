import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from chien_deepfake.datasets import DeepfakeImageFolderDataset, build_transforms
from chien_deepfake.evaluation import evaluate
from chien_deepfake.models import build_model
from chien_deepfake.utils import load_config
from chien_deepfake.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved deepfake detector checkpoint.")
    parser.add_argument("--model", required=True, choices=["efficientb4", "fwa", "ucf", "srm", "spsl"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["ffpp", "celebdf"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_name"] = args.model
    data_cfg = config.get("data", {})
    transform = build_transforms(config, train=False)
    if args.dataset == "ffpp":
        dataset = DeepfakeImageFolderDataset(data_cfg["ffpp_root"], dataset="ffpp", split=args.split, transform=transform)
        output_name = f"{args.model}_ffpp_{args.split}_metrics.json"
    else:
        dataset = DeepfakeImageFolderDataset(data_cfg["celebdf_test_root"], dataset="celebdf", transform=transform)
        output_name = f"{args.model}_celebdf_test_metrics.json"

    counts = dataset.class_counts()
    print(f"{args.dataset.upper()} {args.split}: real={counts['real']} fake={counts['fake']} total={counts['total']}")
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 16)), shuffle=False, num_workers=int(data_cfg.get("num_workers", 4)))

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("cuda", True) else "cpu")
    model = build_model(args.model, config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics, _ = evaluate(model, loader, device)

    results_dir = Path(config.get("evaluation", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / output_name).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(metrics)
    print(f"Saved metrics: {results_dir / output_name}")


if __name__ == "__main__":
    main()

