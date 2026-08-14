"""Minimal training loop for the static-graph weather GNN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import WeatherGraphDataset
from src.models import LossWeights, PIGNNLoss, build_pignn_from_config
from src.training.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PI-GNN on processed monthly tensors.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dynamic-dir", default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--train-months", nargs="*", default=None)
    parser.add_argument("--val-months", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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


def paired_month_files(dynamic_dir: Path, target_dir: Path) -> list[tuple[str, Path, Path]]:
    dynamic = {month_from_path(path): path for path in sorted(dynamic_dir.glob("era5_dynamic_*.pt"))}
    targets = {month_from_path(path): path for path in sorted(target_dir.glob("targets_*.pt"))}
    months = sorted(set(dynamic) & set(targets))
    return [(month, dynamic[month], targets[month]) for month in months]


def split_months(months: list[str], validation_months: int) -> tuple[list[str], list[str]]:
    if validation_months <= 0 or validation_months >= len(months):
        return months, []
    return months[:-validation_months], months[-validation_months:]


def collate_samples(samples: list[dict]) -> dict:
    first = samples[0]
    batch = {
        "x": torch.stack([sample["x"] for sample in samples], dim=0),
        "pos": first["pos"],
        "edge_index": first["edge_index"],
        "edge_attr": first["edge_attr"],
        "timestamp": [sample["timestamp"] for sample in samples],
    }
    if "y" in first:
        batch["y"] = torch.stack([sample["y"] for sample in samples], dim=0)
    return batch


def make_loader(graph_path: Path, dynamic_paths: list[Path], target_paths: list[Path], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = WeatherGraphDataset(graph_path, dynamic_paths, target_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_samples)


def evaluate_epoch(model, loader, loss_fn, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    loss_total = 0.0
    pred_batches = []
    target_batches = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            pos = batch["pos"].to(device)
            prediction = model(x, edge_index, edge_attr)
            losses = loss_fn(prediction, y, edge_index, x[..., 6], pos)
            loss_total += float(losses["total"])
            pred_batches.append(prediction.cpu())
            target_batches.append(y.cpu())
    metrics = summarize_metrics(torch.cat(pred_batches, dim=0), torch.cat(target_batches, dim=0)) if pred_batches else {}
    mean_loss = loss_total / max(len(loader), 1)
    return mean_loss, metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    graph_path = resolve_path(config["paths"]["graph_output"])
    dynamic_dir = resolve_path(args.dynamic_dir or config["era5"]["processed_output_dir"])
    target_dir = resolve_path(args.target_dir or config["targets"]["output_dir"])
    checkpoint_dir = resolve_path(args.checkpoint_dir or config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    month_files = paired_month_files(dynamic_dir, target_dir)
    if not month_files:
        raise ValueError("No paired dynamic/target monthly tensors found")
    available_months = [month for month, _, _ in month_files]
    if args.train_months:
        train_months = args.train_months
        val_months = args.val_months or []
    else:
        train_months, val_months = split_months(available_months, int(config["training"]["validation_months"]))

    train_pairs = [(d, t) for month, d, t in month_files if month in train_months]
    val_pairs = [(d, t) for month, d, t in month_files if month in val_months]
    if not train_pairs:
        raise ValueError("No training months selected")

    batch_size = int(args.batch_size or config["training"]["batch_size"])
    train_loader = make_loader(graph_path, [d for d, _ in train_pairs], [t for _, t in train_pairs], batch_size, True)
    val_loader = (
        make_loader(graph_path, [d for d, _ in val_pairs], [t for _, t in val_pairs], batch_size, False)
        if val_pairs
        else None
    )

    model = build_pignn_from_config(config["model"])
    device = torch.device(args.device or config["training"]["device"])
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(args.learning_rate or config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_cfg = config["loss"]
    loss_fn = PIGNNLoss(
        weights=LossWeights(**loss_cfg["weights"]),
        data_loss=loss_cfg["data_loss"],
        channel_weights=tuple(loss_cfg["channel_weights"]) if loss_cfg["channel_weights"] else None,
        temperature_channel=int(loss_cfg["temperature_channel"]),
        u_channel=int(loss_cfg["u_channel"]),
        v_channel=int(loss_cfg["v_channel"]),
        lapse_rate_k_per_m=float(loss_cfg["lapse_rate_k_per_m"]),
    )

    epochs = int(args.epochs or config["training"]["epochs"])
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            pos = batch["pos"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, edge_index, edge_attr)
            losses = loss_fn(prediction, y, edge_index, x[..., 6], pos)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            running += float(losses["total"].detach())

        train_loss = running / max(len(train_loader), 1)
        print(f"epoch {epoch} train_loss={train_loss:.6f}")

        val_loss = train_loss
        if val_loader is not None:
            val_loss, metrics = evaluate_epoch(model, val_loader, loss_fn, device)
            print(f"epoch {epoch} val_loss={val_loss:.6f} metrics={metrics}")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "train_months": train_months,
            "val_months": val_months,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
