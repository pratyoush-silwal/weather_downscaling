#!/usr/bin/env python3
"""Generate a Markdown comparison report with plots, without Jupyter."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained weather models without Jupyter.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report", default="notebooks/cloud/compare_weather_models_report.md")
    parser.add_argument("--output-dir", default="notebooks/cloud/artifacts")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--months", nargs="*", default=None)
    return parser.parse_args()


ROOT = repo_root()

import sys

sys.path.insert(0, str(ROOT))

from src.data.dataset import WeatherGraphDataset
from src.models import build_mlp_baseline_from_config, build_pignn_from_config
from src.training.train import collate_samples, default_split_months, resolve_device


TARGET_NAMES = ["temperature", "precipitation", "u_wind", "v_wind"]
EVENT_THRESHOLD = 0.1


def month_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def _expanded_node_mask(node_mask: torch.Tensor | None, y_true: torch.Tensor) -> torch.Tensor | None:
    if node_mask is None:
        return None
    node_mask = node_mask.to(dtype=torch.bool)
    if node_mask.ndim == y_true.ndim - 2:
        while node_mask.ndim < y_true.ndim - 1:
            node_mask = node_mask.unsqueeze(0)
        return node_mask.unsqueeze(-1).expand_as(y_true)
    if node_mask.ndim == y_true.ndim - 1:
        return node_mask.unsqueeze(-1).expand_as(y_true)
    if node_mask.ndim == y_true.ndim and node_mask.shape == y_true.shape:
        return node_mask
    raise ValueError(f"node_mask shape {tuple(node_mask.shape)} is incompatible with target shape {tuple(y_true.shape)}")


def _finite_pair(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    region_mask = _expanded_node_mask(node_mask, y_true)
    if region_mask is not None:
        mask = mask & region_mask
    return y_true[mask], y_pred[mask]


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred, node_mask=node_mask)
    return torch.sqrt(((y_pred - y_true) ** 2).mean()).item()


def mae(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred, node_mask=node_mask)
    return (y_pred - y_true).abs().mean().item()


def bias(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred, node_mask=node_mask)
    return (y_pred - y_true).mean().item()


def corr(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred, node_mask=node_mask)
    if y_true.numel() < 2:
        return float("nan")
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = torch.sqrt((y_true**2).sum() * (y_pred**2).sum())
    return ((y_true * y_pred).sum() / denom).item() if denom > 0 else float("nan")


def r2(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred, node_mask=node_mask)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")


def nse(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> float:
    return r2(y_true, y_pred, node_mask=node_mask)


def kge(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred, node_mask=node_mask)
    r = corr(y_true, y_pred)
    mean_true = y_true.mean().item()
    mean_pred = y_pred.mean().item()
    std_true = y_true.std(unbiased=False).item()
    std_pred = y_pred.std(unbiased=False).item()
    alpha = std_pred / std_true if std_true else float("nan")
    beta = mean_pred / mean_true if mean_true else float("nan")
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def precip_event_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = EVENT_THRESHOLD, node_mask: torch.Tensor | None = None) -> dict[str, float]:
    y_true, y_pred = _finite_pair(y_true, y_pred, node_mask=node_mask)
    truth = y_true >= threshold
    pred = y_pred >= threshold
    tp = (truth & pred).sum().item()
    fp = ((~truth) & pred).sum().item()
    fn = (truth & (~pred)).sum().item()
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    fbias = (tp + fp) / (tp + fn) if (tp + fn) else float("nan")
    return {"pod": pod, "far": far, "csi": csi, "frequency_bias": fbias}


def metrics_for_channel(y_true: torch.Tensor, y_pred: torch.Tensor, is_precip: bool = False, node_mask: torch.Tensor | None = None) -> dict[str, float]:
    out = {
        "rmse": rmse(y_true, y_pred, node_mask=node_mask),
        "mae": mae(y_true, y_pred, node_mask=node_mask),
        "bias": bias(y_true, y_pred, node_mask=node_mask),
        "corr": corr(y_true, y_pred, node_mask=node_mask),
        "r2": r2(y_true, y_pred, node_mask=node_mask),
        "nse": nse(y_true, y_pred, node_mask=node_mask),
        "kge": kge(y_true, y_pred, node_mask=node_mask),
    }
    if is_precip:
        out.update(precip_event_metrics(y_true, y_pred, node_mask=node_mask))
    return out


def summarize_all(y_true: torch.Tensor, y_pred: torch.Tensor, node_mask: torch.Tensor | None = None) -> pd.DataFrame:
    rows = {}
    for idx, name in enumerate(TARGET_NAMES):
        channel_mask = node_mask if node_mask is None else node_mask
        rows[name] = metrics_for_channel(y_true[..., idx], y_pred[..., idx], is_precip=(name == "precipitation"), node_mask=channel_mask)
    return pd.DataFrame.from_dict(rows, orient="index")


def artifact_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def require_artifact(path: Path, model_name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact for {model_name}: {path}")
    return path


def load_mlp(device: torch.device, config: dict[str, Any]):
    model = build_mlp_baseline_from_config(config["baselines"]["mlp"])
    state_path = require_artifact(artifact_path(config["baselines"]["output_dir"], "mlp", "model.pt"), "mlp")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_xgboost(config: dict[str, Any]):
    model_path = require_artifact(artifact_path(config["baselines"]["output_dir"], "xgboost", "model.pkl"), "xgboost")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def load_gnn(device: torch.device, config: dict[str, Any]):
    model = build_pignn_from_config(config["model"])
    checkpoint_path = require_artifact(artifact_path(config["training"]["checkpoint_dir"], "best.pt"), "gnn")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_interpolation(batch: dict[str, torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    idx = config["baselines"]["interpolation"]["coarse_feature_indices"]
    return torch.stack([batch["x"][..., i] for i in idx], dim=-1)


def collect_predictions(
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = []
    ys = []
    if model_name == "interpolation":
        for batch in loader:
            preds.append(predict_interpolation(batch, config))
            ys.append(batch["y"])
        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

    if model_name == "mlp":
        model = load_mlp(device, config)
        with torch.no_grad():
            for batch in loader:
                preds.append(model(batch["x"].to(device)).cpu())
                ys.append(batch["y"])
        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

    if model_name == "xgboost":
        model = load_xgboost(config)
        for batch in loader:
            x = batch["x"].reshape(-1, batch["x"].shape[-1]).numpy()
            y = batch["y"]
            pred = torch.from_numpy(model.predict(x)).reshape_as(y)
            preds.append(pred)
            ys.append(y)
        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

    if model_name == "gnn":
        model = load_gnn(device, config)
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["x"].to(device), batch["edge_index"].to(device), batch["edge_attr"].to(device)).cpu()
                preds.append(pred)
                ys.append(batch["y"])
        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

    raise ValueError(model_name)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def write_markdown(
    path: Path,
    split: str,
    months: list[str],
    summary: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    images: list[tuple[str, Path]],
    skipped_models: dict[str, str],
) -> None:
    artifact_dir = path.parent
    lines = [
        "# Weather Model Comparison",
        "",
        "## Evaluation Setup",
        "",
        f"- split: `{split}`",
        f"- months: `{months[0]}` to `{months[-1]}` ({len(months)} monthly files)",
        f"- skipped models: `{list(skipped_models)}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    for model_name, table in metric_tables.items():
        lines.extend([f"## {model_name}", "", table.to_markdown(), ""])

    for title, image_path in images:
        relative = image_path.relative_to(artifact_dir)
        lines.extend([f"## {title}", "", f"![{title}]({relative.as_posix()})", ""])

    if skipped_models:
        lines.extend(["## Skipped Models", "", code_block(json.dumps(skipped_models, indent=2), "json"), ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def code_block(text: str, language: str = "") -> str:
    fence = f"```{language}".rstrip()
    return f"{fence}\n{text}\n```"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["training"]["device"])
    output_dir = resolve_path(args.output_dir)
    report_path = resolve_path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    graph_path = resolve_path(config["paths"]["graph_output"])
    dynamic_dir = resolve_path(config["era5"]["processed_output_dir"])
    target_dir = resolve_path(config["targets"]["output_dir"])

    dynamic = {month_from_path(path): path for path in sorted(dynamic_dir.glob("era5_dynamic_*.pt"))}
    targets = {month_from_path(path): path for path in sorted(target_dir.glob("targets_*.pt"))}
    months = sorted(set(dynamic) & set(targets))
    if args.months:
        months = [month for month in months if month in set(args.months)]
    else:
        train_months, val_months, test_months = default_split_months(config, months)
        if args.split == "train":
            months = train_months
        elif args.split == "val":
            months = val_months
        elif args.split == "test":
            months = test_months
    assert months, "No paired months found."

    dataset = WeatherGraphDataset(graph_path, [dynamic[m] for m in months], [targets[m] for m in months])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)
    region_mask = dataset.in_region_mask

    requested_models = ["interpolation", "mlp", "xgboost", "gnn"]
    available_models: list[str] = []
    predictions: dict[str, torch.Tensor] = {}
    metric_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | str]] = []
    reference_target: torch.Tensor | None = None
    skipped_models: dict[str, str] = {}

    for model_name in requested_models:
        try:
            pred, y_true = collect_predictions(model_name, loader, device, config)
        except FileNotFoundError as exc:
            skipped_models[model_name] = str(exc)
            continue
        available_models.append(model_name)
        predictions[model_name] = pred
        if reference_target is None:
            reference_target = y_true
        table = summarize_all(y_true, pred, node_mask=region_mask)
        metric_tables[model_name] = table
        summary_rows.append(
            {
                "model": model_name,
                "mean_rmse": float(table["rmse"].mean()),
                "mean_mae": float(table["mae"].mean()),
                "mean_kge": float(table["kge"].mean()),
                "precip_csi": float(table.loc["precipitation", "csi"]),
            }
        )

    assert summary_rows, "No model artifacts available for comparison."
    assert reference_target is not None
    summary = pd.DataFrame(summary_rows).sort_values("mean_rmse")
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2)
    for model_name, table in metric_tables.items():
        table.to_csv(output_dir / f"{model_name}_metrics.csv")

    images: list[tuple[str, Path]] = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    summary.plot.bar(x="model", y="mean_rmse", ax=axes[0], legend=False, title="Mean RMSE")
    summary.plot.bar(x="model", y="mean_mae", ax=axes[1], legend=False, title="Mean MAE")
    summary.plot.bar(x="model", y="mean_kge", ax=axes[2], legend=False, title="Mean KGE")
    for ax in axes:
        ax.grid(alpha=0.3)
    path = output_dir / "summary_metrics.png"
    save_fig(path)
    images.append(("Summary Metrics", path))

    rmse_rows = []
    for model_name, table in metric_tables.items():
        for target, row in table.iterrows():
            rmse_rows.append({"model": model_name, "target": target, "rmse": row["rmse"]})
    pivot = pd.DataFrame(rmse_rows).pivot(index="target", columns="model", values="rmse")
    pivot.plot.bar(figsize=(10, 5), grid=True, title="RMSE by Target and Model")
    path = output_dir / "rmse_by_target.png"
    save_fig(path)
    images.append(("RMSE by Target", path))

    model_names = available_models
    sample_step = max(1, len(dataset) // 200)
    flat_true = reference_target[::sample_step].reshape(-1, reference_target.shape[-1])
    fig, axes = plt.subplots(len(TARGET_NAMES), len(model_names), figsize=(4 * len(model_names), 3 * len(TARGET_NAMES)), squeeze=False)
    for col, model_name in enumerate(model_names):
        flat_pred = predictions[model_name][::sample_step].reshape(-1, predictions[model_name].shape[-1])
        for row, target_name in enumerate(TARGET_NAMES):
            ax = axes[row, col]
            ax.scatter(flat_true[:, row].numpy(), flat_pred[:, row].numpy(), s=4, alpha=0.2)
            ax.set_title(f"{model_name} | {target_name}")
            ax.set_xlabel("target")
            ax.set_ylabel("pred")
            ax.grid(alpha=0.2)
    path = output_dir / "scatter_grid.png"
    save_fig(path)
    images.append(("Prediction Scatter Grid", path))

    node_idx = 0
    time_steps = min(168, len(dataset))
    fig, axes = plt.subplots(len(TARGET_NAMES), 1, figsize=(14, 10), sharex=True)
    for row, target_name in enumerate(TARGET_NAMES):
        axes[row].plot(reference_target[:time_steps, node_idx, row].numpy(), label="target", linewidth=2)
        for model_name in model_names:
            axes[row].plot(predictions[model_name][:time_steps, node_idx, row].numpy(), label=model_name, alpha=0.8)
        axes[row].set_title(target_name)
        axes[row].grid(alpha=0.3)
    axes[0].legend(ncol=max(1, len(model_names) + 1), fontsize=8)
    path = output_dir / "timeseries_node0.png"
    save_fig(path)
    images.append(("Short Time Series", path))

    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 3), sharey=True, squeeze=False)
    for ax, model_name in zip(axes[0], model_names):
        residual = (predictions[model_name][..., 1] - reference_target[..., 1]).reshape(-1).numpy()
        ax.hist(residual, bins=60, alpha=0.8)
        ax.set_title(f"{model_name} precip residual")
        ax.grid(alpha=0.2)
    path = output_dir / "precip_residuals.png"
    save_fig(path)
    images.append(("Precipitation Residuals", path))

    write_markdown(report_path, args.split, months, summary, metric_tables, images, skipped_models)
    print(report_path)


if __name__ == "__main__":
    main()
