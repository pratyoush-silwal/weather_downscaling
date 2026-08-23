"""Train or evaluate simple baselines against the same monthly tensors."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
import sys

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import WeatherGraphDataset
from src.models import build_mlp_baseline_from_config
from src.training.metrics import summarize_metrics
from src.training.train import collate_samples, default_split_months, load_config, paired_month_files, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interpolation, MLP, or XGBoost baselines.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["interpolation", "mlp", "xgboost"], required=True)
    parser.add_argument("--dynamic-dir", default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-months", nargs="*", default=None)
    parser.add_argument("--val-months", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_month_datasets(config: dict, dynamic_dir: Path, target_dir: Path, train_months: list[str] | None, val_months: list[str] | None):
    graph_path = resolve_path(config["paths"]["graph_output"])
    month_files = paired_month_files(dynamic_dir, target_dir)
    if not month_files:
        raise ValueError("No paired dynamic/target monthly tensors found")
    months = [month for month, _, _ in month_files]
    if train_months is None:
        train_months, val_months, test_months = default_split_months(config, months)
    else:
        test_months = [month for month in months if month not in set(train_months) | set(val_months or [])]
    train_pairs = [(d, t) for month, d, t in month_files if month in train_months]
    val_pairs = [(d, t) for month, d, t in month_files if month in (val_months or [])]
    test_pairs = [(d, t) for month, d, t in month_files if month in test_months]
    train_dataset = WeatherGraphDataset(graph_path, [d for d, _ in train_pairs], [t for _, t in train_pairs])
    val_dataset = WeatherGraphDataset(graph_path, [d for d, _ in val_pairs], [t for _, t in val_pairs]) if val_pairs else None
    test_dataset = WeatherGraphDataset(graph_path, [d for d, _ in test_pairs], [t for _, t in test_pairs]) if test_pairs else None
    return train_dataset, val_dataset, test_dataset, train_months, val_months or [], test_months


def evaluate_model(model: nn.Module, dataset: WeatherGraphDataset, device: torch.device) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_samples)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            preds.append(model(x).cpu())
            targets.append(batch["y"].cpu())
    return summarize_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def evaluate_xgboost_model(model, dataset: WeatherGraphDataset) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)
    preds = []
    targets = []
    for batch in loader:
        x = batch["x"].reshape(-1, batch["x"].shape[-1]).numpy()
        y = batch["y"]
        pred = torch.from_numpy(model.predict(x)).reshape_as(y)
        preds.append(pred)
        targets.append(y)
    return summarize_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def interpolation_predict(x: torch.Tensor, coarse_indices: list[int]) -> torch.Tensor:
    return torch.stack([x[..., index] for index in coarse_indices], dim=-1)


class InterpolationWrapper(nn.Module):
    def __init__(self, coarse_indices: list[int]) -> None:
        super().__init__()
        self.coarse_indices = coarse_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return interpolation_predict(x, self.coarse_indices)


def train_mlp(config: dict, train_dataset: WeatherGraphDataset, val_dataset: WeatherGraphDataset | None, device: torch.device):
    model = build_mlp_baseline_from_config(config["baselines"]["mlp"])
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["baselines"]["mlp"]["learning_rate"]),
        weight_decay=float(config["baselines"]["mlp"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["baselines"]["mlp"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_samples,
    )
    for _ in range(int(config["baselines"]["mlp"]["epochs"])):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["x"].to(device))
            loss = loss_fn(prediction, batch["y"].to(device))
            loss.backward()
            optimizer.step()
    train_metrics = evaluate_model(model, train_dataset, device)
    val_metrics = evaluate_model(model, val_dataset, device) if val_dataset is not None else {}
    return model, train_metrics, val_metrics


def _sample_row_indices(length: int, max_rows: int, rng: random.Random) -> list[int]:
    if length <= max_rows:
        return list(range(length))
    return rng.sample(range(length), max_rows)


def rows_from_dataset(dataset: WeatherGraphDataset, max_rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    indices = _sample_row_indices(len(dataset), max_rows, rng)
    # ponytail: sample timesteps, not every node-time row; increase max_rows if xgboost underfits.
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)
    features = []
    targets = []
    for batch in loader:
        features.append(batch["x"].reshape(-1, batch["x"].shape[-1]))
        targets.append(batch["y"].reshape(-1, batch["y"].shape[-1]))
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def train_xgboost(config: dict, train_dataset: WeatherGraphDataset, val_dataset: WeatherGraphDataset | None):
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
    except Exception as exc:
        raise SystemExit("xgboost baseline needs `pip install xgboost`.") from exc

    cfg = config["baselines"]["xgboost"]
    x_train, y_train = rows_from_dataset(train_dataset, int(cfg["max_train_rows"]), int(cfg["random_state"]))
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            learning_rate=float(cfg["learning_rate"]),
            subsample=float(cfg["subsample"]),
            colsample_bytree=float(cfg["colsample_bytree"]),
            reg_lambda=float(cfg["reg_lambda"]),
            random_state=int(cfg["random_state"]),
            tree_method="hist",
            objective="reg:squarederror",
        )
    )
    model.fit(x_train.numpy(), y_train.numpy())

    train_metrics = evaluate_xgboost_model(model, train_dataset)
    val_metrics = evaluate_xgboost_model(model, val_dataset) if val_dataset is not None else {}
    return model, train_metrics, val_metrics


def mean_rmse(metrics: dict[str, float]) -> float:
    keys = [key for key in metrics if key.endswith("_rmse")]
    return sum(metrics[key] for key in keys) / len(keys) if keys else float("nan")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dynamic_dir = resolve_path(args.dynamic_dir or config["era5"]["processed_output_dir"])
    target_dir = resolve_path(args.target_dir or config["targets"]["output_dir"])
    output_dir = resolve_path(args.output_dir or config["baselines"]["output_dir"]) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, train_months, val_months, test_months = build_month_datasets(
        config, dynamic_dir, target_dir, args.train_months, args.val_months
    )

    if args.model == "interpolation":
        model = InterpolationWrapper(list(config["baselines"]["interpolation"]["coarse_feature_indices"]))
        train_metrics = evaluate_model(model, train_dataset, torch.device("cpu"))
        val_metrics = evaluate_model(model, val_dataset, torch.device("cpu")) if val_dataset is not None else {}
        test_metrics = evaluate_model(model, test_dataset, torch.device("cpu")) if test_dataset is not None else {}
        artifact = None
    elif args.model == "mlp":
        model, train_metrics, val_metrics = train_mlp(
            config,
            train_dataset,
            val_dataset,
            torch.device(args.device or config["training"]["device"]),
        )
        test_metrics = (
            evaluate_model(model, test_dataset, torch.device(args.device or config["training"]["device"]))
            if test_dataset is not None
            else {}
        )
        artifact = output_dir / "model.pt"
        torch.save(model.state_dict(), artifact)
    else:
        model, train_metrics, val_metrics = train_xgboost(config, train_dataset, val_dataset)
        test_metrics = evaluate_xgboost_model(model, test_dataset) if test_dataset is not None else {}
        artifact = output_dir / "model.pkl"
        with artifact.open("wb") as handle:
            pickle.dump(model, handle)

    metrics = {
        "model": args.model,
        "train_months": train_months,
        "val_months": val_months,
        "test_months": test_months,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "primary_metric": config["baselines"]["primary_metric"],
        "validation_primary_value": mean_rmse(val_metrics) if val_metrics else mean_rmse(train_metrics),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(metrics)
    if artifact is not None:
        print(f"saved {artifact}")


if __name__ == "__main__":
    main()
