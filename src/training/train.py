"""Minimal training loop for the static-graph weather GNN."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import WeatherGraphDataset, MonthlySampler
from src.models import LossWeights, PIGNNLoss, build_pignn_from_config
from src.training.metrics import StreamingMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PI-GNN on processed monthly tensors.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--dynamic-dir", default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--train-months", nargs="*", default=None)
    parser.add_argument("--val-months", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_name: str | None) -> torch.device:
    name = str(device_name or "auto").lower()
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def month_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def paired_month_files(dynamic_dir: Path, target_dir: Path) -> list[tuple[str, Path, Path]]:
    dynamic = {month_from_path(path): path for path in sorted(dynamic_dir.glob("era5_dynamic_*.pt"))}
    targets = {month_from_path(path): path for path in sorted(target_dir.glob("targets_*.pt"))}
    months = sorted(set(dynamic) & set(targets))
    return [(month, dynamic[month], targets[month]) for month in months]


def year_from_month(month: str) -> int:
    return int(month[:4])


def split_months_by_year(
    months: list[str],
    train_years: int,
    validation_years: int,
    test_years: int,
) -> tuple[list[str], list[str], list[str]]:
    years = sorted({year_from_month(month) for month in months})
    total = train_years + validation_years + test_years
    if total <= 0:
        raise ValueError("Split sizes must be positive")
    if total > len(years):
        available = len(years)
        test_years = min(test_years, max(available - 2, 0))
        validation_years = min(validation_years, max(available - test_years - 1, 0))
        train_years = max(available - validation_years - test_years, 1)
        total = train_years + validation_years + test_years
    selected_years = years[-total:]
    train_set = set(selected_years[:train_years])
    val_set = set(selected_years[train_years : train_years + validation_years])
    test_set = set(selected_years[train_years + validation_years :])
    train_months = [month for month in months if year_from_month(month) in train_set]
    val_months = [month for month in months if year_from_month(month) in val_set]
    test_months = [month for month in months if year_from_month(month) in test_set]
    return train_months, val_months, test_months


def default_split_months(config: dict, months: list[str]) -> tuple[list[str], list[str], list[str]]:
    training = config["training"]
    return split_months_by_year(
        months,
        int(training["train_years"]),
        int(training["validation_years"]),
        int(training["test_years"]),
    )


def collate_samples(samples: list[dict]) -> dict:
    first = samples[0]
    batch = {
        "x": torch.stack([sample["x"] for sample in samples], dim=0),
        "pos": first["pos"],
        "in_region_mask": first["in_region_mask"],
        "edge_index": first["edge_index"],
        "edge_attr": first["edge_attr"],
        "timestamp": [sample["timestamp"] for sample in samples],
    }
    if "y" in first:
        batch["y"] = torch.stack([sample["y"] for sample in samples], dim=0)
    return batch


def make_loader(graph_path: Path, dynamic_paths: list[Path], target_paths: list[Path], batch_size: int, shuffle: bool, workers: int = 0) -> DataLoader:
    dataset = WeatherGraphDataset(graph_path, dynamic_paths, target_paths)
    return DataLoader(dataset, batch_size=batch_size, sampler=MonthlySampler(dataset) if shuffle else None, num_workers=workers, collate_fn=collate_samples)


def evaluate_epoch(model, loader, loss_fn, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    loss_total = 0.0
    accumulator = StreamingMetrics()
    edge_index, edge_attr, pos = (getattr(loader.dataset, k).to(device) for k in ("edge_index", "edge_attr", "pos"))
    region = loader.dataset.in_region_mask.to(device)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            prediction = model(x, edge_index, edge_attr)
            losses = loss_fn(prediction, y, edge_index, x[..., 6], pos)
            loss_total += losses["total"].detach().double()
            accumulator.update(prediction, y, region)
    metrics = accumulator.compute()
    mean_loss = float(loss_total) / max(len(loader), 1)
    return mean_loss, metrics


def effective_config(config: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = float(args.learning_rate)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    return cfg


def append_history_row(path: Path, row: dict[str, object]) -> None:
    fieldnames = list(row)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(state, path):
    temporary = path.with_suffix(".pt.part")
    torch.save(state, temporary)
    temporary.replace(path)


def write_summary(
    path: Path,
    best_epoch: int,
    best_val: float,
    best_metrics: dict[str, float],
    last_epoch: int,
    train_months: list[str],
    val_months: list[str],
    test_months: list[str],
    graph_path: Path,
    dynamic_dir: Path,
    target_dir: Path,
) -> None:
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_metrics": best_metrics,
        "last_epoch": last_epoch,
        "train_months": train_months,
        "val_months": val_months,
        "test_months": test_months,
        "graph": str(graph_path),
        "dynamic_dir": str(dynamic_dir),
        "target_dir": str(target_dir),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config = effective_config(base_config, args)
    graph_path = resolve_path(args.graph or config["paths"]["graph_output"])
    dynamic_dir = resolve_path(args.dynamic_dir or config["era5"]["processed_output_dir"])
    target_dir = resolve_path(args.target_dir or config["targets"]["output_dir"])
    checkpoint_dir = resolve_path(args.checkpoint_dir or config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.csv"
    metrics_path = checkpoint_dir / "metrics.json"
    with (checkpoint_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    month_files = paired_month_files(dynamic_dir, target_dir)
    if not month_files:
        raise ValueError("No paired dynamic/target monthly tensors found")
    available_months = [month for month, _, _ in month_files]
    if args.train_months:
        train_months = args.train_months
        val_months = args.val_months or []
        test_months = [month for month in available_months if month not in set(train_months) | set(val_months)]
    else:
        train_months, val_months, test_months = default_split_months(config, available_months)

    train_pairs = [(d, t) for month, d, t in month_files if month in train_months]
    val_pairs = [(d, t) for month, d, t in month_files if month in val_months]
    if not train_pairs:
        raise ValueError("No training months selected")

    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"].get("num_workers", 0))
    train_loader = make_loader(graph_path, [d for d, _ in train_pairs], [t for _, t in train_pairs], batch_size, True, workers)
    val_loader = (
        make_loader(graph_path, [d for d, _ in val_pairs], [t for _, t in val_pairs], batch_size, False, workers)
        if val_pairs
        else None
    )

    model = build_pignn_from_config(config["model"])
    device = resolve_device(config["training"]["device"])
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_cfg = config["loss"]
    loss_fn = PIGNNLoss(
        weights=LossWeights(**loss_cfg["weights"]),
        data_loss=loss_cfg["data_loss"],
        channel_scales=loss_cfg.get("channel_scales"),
        channel_weights=tuple(loss_cfg["channel_weights"]) if loss_cfg["channel_weights"] else None,
        temperature_channel=int(loss_cfg["temperature_channel"]),
        u_channel=int(loss_cfg["u_channel"]),
        v_channel=int(loss_cfg["v_channel"]),
        lapse_rate_k_per_m=float(loss_cfg["lapse_rate_k_per_m"]),
    )

    epochs = int(config["training"]["epochs"])
    last_checkpoint_path = checkpoint_dir / "last.pt"
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    progress = {}
    interval = int(config["training"].get("recovery_steps", 0))
    bf16 = bool(config["training"].get("cuda_bfloat16", False))
    if bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("cuda_bfloat16 requires a GPU with bfloat16 support")
    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        progress = checkpoint.get("progress", {})
        if progress and (checkpoint["train_months"] != train_months or checkpoint["val_months"] != val_months
                         or checkpoint["config"]["training"]["batch_size"] != batch_size):
            raise ValueError("Cannot resume partial epoch with a different split or batch size")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metrics = dict(checkpoint.get("best_metrics", {}))
        print(f"resuming from epoch {start_epoch}")

    if start_epoch > epochs:
        print(f"training already finished at epoch {start_epoch - 1}")
        write_summary(
            metrics_path,
            best_epoch,
            best_val,
            best_metrics,
            start_epoch - 1,
            train_months,
            val_months,
            test_months,
            graph_path,
            dynamic_dir,
            target_dir,
        )
        return

    edge_index, edge_attr, pos = (getattr(train_loader.dataset, k).to(device) for k in ("edge_index", "edge_attr", "pos"))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = progress.get("running", 0.0)
        step = progress.get("step", 0)
        train_loader.sampler.seed = progress["seed"] if progress else int(torch.randint(2**31, ()).item())
        train_loader.sampler.skip = step * batch_size
        iterator = iter(train_loader)
        if progress:
            torch.set_rng_state(progress["cpu_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(progress["cuda_rng"], device)
        progress = {}
        for batch in iterator:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                prediction = model(x, edge_index, edge_attr)
            prediction = prediction.float()
            losses = loss_fn(prediction, y, edge_index, x[..., 6], pos)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            running += losses["total"].detach().double()
            step += 1
            if interval > 0 and step % interval == 0:
                save_checkpoint(dict(epoch=epoch-1, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                                     config=config, train_months=train_months, val_months=val_months, test_months=test_months,
                                     best_val_loss=best_val, best_epoch=best_epoch, best_metrics=best_metrics,
                                     progress=dict(step=step, seed=train_loader.sampler.seed, running=float(running),
                                                   cpu_rng=torch.get_rng_state(),
                                                   cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)),
                                last_checkpoint_path)

        train_loss = float(running) / max(len(train_loader), 1)
        print(f"epoch {epoch} train_loss={train_loss:.6f}")

        val_loss = train_loss
        metrics: dict[str, float] = {}
        if val_loader is not None:
            val_loss, metrics = evaluate_epoch(model, val_loader, loss_fn, device)
            print(f"epoch {epoch} val_loss={val_loss:.6f} metrics={metrics}")

        history_row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        for key, value in metrics.items():
            history_row[key] = value
        append_history_row(history_path, history_row)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "train_months": train_months,
            "val_months": val_months,
            "test_months": test_months,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        }
        if val_loss <= best_val:
            best_val = val_loss
            best_epoch = epoch
            best_metrics = metrics
            checkpoint["best_val_loss"] = best_val
            checkpoint["best_epoch"] = best_epoch
            checkpoint["best_metrics"] = best_metrics
            save_checkpoint(checkpoint, checkpoint_dir / "best.pt")
        save_checkpoint(checkpoint, checkpoint_dir / "last.pt")

    write_summary(
        metrics_path,
        best_epoch,
        best_val,
        best_metrics,
        epochs,
        train_months,
        val_months,
        test_months,
        graph_path,
        dynamic_dir,
        target_dir,
    )


if __name__ == "__main__":
    main()
