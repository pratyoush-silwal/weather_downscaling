"""Evaluate a trained PI-GNN checkpoint on processed monthly tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import WeatherGraphDataset
from src.models import build_pignn_from_config
from src.training.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PI-GNN checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--dynamic-dir", default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--months", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def month_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def collate_samples(samples: list[dict]) -> dict:
    first = samples[0]
    return {
        "x": torch.stack([sample["x"] for sample in samples], dim=0),
        "y": torch.stack([sample["y"] for sample in samples], dim=0),
        "pos": first["pos"],
        "edge_index": first["edge_index"],
        "edge_attr": first["edge_attr"],
        "timestamp": [sample["timestamp"] for sample in samples],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    graph_path = resolve_path(args.graph or config["paths"]["graph_output"])
    dynamic_dir = resolve_path(args.dynamic_dir or config["era5"]["processed_output_dir"])
    target_dir = resolve_path(args.target_dir or config["targets"]["output_dir"])

    dynamic = {month_from_path(path): path for path in sorted(dynamic_dir.glob("era5_dynamic_*.pt"))}
    targets = {month_from_path(path): path for path in sorted(target_dir.glob("targets_*.pt"))}
    months = sorted(set(dynamic) & set(targets))
    if args.months:
        months = [month for month in months if month in set(args.months)]
    if not months:
        raise ValueError("No evaluation months selected")

    dataset = WeatherGraphDataset(
        graph_path,
        [dynamic[month] for month in months],
        [targets[month] for month in months],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)

    checkpoint = torch.load(resolve_path(args.checkpoint), map_location="cpu", weights_only=False)
    model = build_pignn_from_config(config["model"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device or config["training"]["device"])
    model.to(device)
    model.eval()

    predictions = []
    targets_out = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["edge_attr"].to(device),
            )
            predictions.append(prediction.cpu())
            targets_out.append(batch["y"].cpu())

    metrics = summarize_metrics(torch.cat(predictions, dim=0), torch.cat(targets_out, dim=0))
    print(metrics)


if __name__ == "__main__":
    main()
