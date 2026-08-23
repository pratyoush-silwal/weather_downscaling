#!/usr/bin/env python3
"""Grid-search hyperparameters without mutating the default config."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.run_buffer_experiment import buffer_tag, repo_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable hyperparameter grid for the GNN.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sweep-config", default="configs/hyperparameter_sweep.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_dotted(config: dict, key: str, value) -> None:
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def get_dotted(config: dict, key: str, default=None):
    target = config
    for part in key.split("."):
        if not isinstance(target, dict) or part not in target:
            return default
        target = target[part]
    return target


def sanitize(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}".replace("-", "m").replace(".", "p")
    return str(value).replace("/", "_").replace("-", "m").replace(".", "p")


def combo_iter(search_space: dict[str, list]) -> list[dict[str, object]]:
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]


def name_for(index: int, overrides: dict[str, object], aliases: dict[str, str]) -> str:
    parts = [f"run_{index:04d}"]
    for key, value in overrides.items():
        alias = aliases.get(key, key.split(".")[-1])
        parts.append(f"{alias}_{sanitize(value)}")
    return "__".join(parts)


def prep_overrides(overrides: dict[str, object], prefixes: list[str]) -> dict[str, object]:
    out = {}
    for key, value in overrides.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            out[key] = value
    return out


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=repo_root(), check=True)


def metrics_value(metrics: dict, key: str):
    value = metrics
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def write_summary(runs_root: Path, selection_metric: str) -> None:
    rows = []
    for metrics_path in sorted(runs_root.glob("run_*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        config_path = metrics_path.parent / "config.yaml"
        config = load_yaml(config_path) if config_path.exists() else {}
        row = {
            "run_name": metrics_path.parent.name,
            "selection_metric": metrics_value(metrics, selection_metric),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_epoch": metrics.get("best_epoch"),
            "clip_buffer_deg": get_dotted(config, "region.clip_buffer_deg"),
            "hidden_channels": get_dotted(config, "model.hidden_channels"),
            "num_layers": get_dotted(config, "model.num_layers"),
            "dropout": get_dotted(config, "model.dropout"),
            "learning_rate": get_dotted(config, "training.learning_rate"),
            "weight_decay": get_dotted(config, "training.weight_decay"),
            "batch_size": get_dotted(config, "training.batch_size"),
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["selection_metric"] is None, row["selection_metric"]))
    with (runs_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        with (runs_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.config)
    sweep = load_yaml(args.sweep_config)

    runs_root = resolve_path(sweep.get("runs_root", "runs/hyperparameter_sweep"))
    runs_root.mkdir(parents=True, exist_ok=True)
    search_space = sweep.get("search_space", {})
    fixed_overrides = sweep.get("fixed_overrides", {})
    aliases = sweep.get("aliases", {})
    prep_prefixes = sweep.get("prep_key_prefixes", ["region.", "graph."])
    selection_metric = sweep.get("selection_metric", "best_val_loss")
    resume = bool(sweep.get("resume", True))
    overwrite_data = bool(sweep.get("overwrite_data", False))
    continue_on_error = bool(sweep.get("continue_on_error", False))
    months = sweep.get("months")
    train_months = sweep.get("train_months")
    val_months = sweep.get("val_months")

    combinations = combo_iter(search_space)
    if args.limit is not None:
        combinations = combinations[: args.limit]

    for index, combo in enumerate(combinations, start=1):
        merged = copy.deepcopy(base_config)
        for key, value in fixed_overrides.items():
            set_dotted(merged, key, value)
        for key, value in combo.items():
            set_dotted(merged, key, value)

        run_name = name_for(index, combo, aliases)
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        status_path = run_dir / "status.json"
        config_path = run_dir / "config.yaml"

        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(merged, handle, sort_keys=False)

        if resume and metrics_path.exists():
            print(f"Skipping completed {run_name}")
            continue

        prep_cfg = prep_overrides({**fixed_overrides, **combo}, prep_prefixes)
        clip_buffer = float(get_dotted(merged, "region.clip_buffer_deg", 0.0))
        data_tag_parts = prep_cfg or {"region.clip_buffer_deg": clip_buffer}
        data_tag = "__".join(
            f"{aliases.get(key, key.split('.')[-1])}_{sanitize(value)}" for key, value in data_tag_parts.items()
        )
        experiment_tag = f"data__{data_tag}"
        experiments_root = resolve_path(merged["training"].get("experiments_root", "data/processed/buffer_experiments"))
        data_root = experiments_root / experiment_tag

        with status_path.open("w", encoding="utf-8") as handle:
            json.dump({"status": "running", "run_name": run_name}, handle, indent=2)

        try:
            prepare_cmd = [
                sys.executable,
                "src/training/run_buffer_experiment.py",
                "--config",
                str(config_path),
                "--clip-buffer-deg",
                str(clip_buffer),
                "--experiment-tag",
                experiment_tag,
            ]
            if months:
                prepare_cmd.extend(["--months", *months])
            if overwrite_data:
                prepare_cmd.append("--overwrite")
            run(prepare_cmd, args.dry_run)

            train_cmd = [
                sys.executable,
                "src/training/train.py",
                "--config",
                str(config_path),
                "--graph",
                str(data_root / "graph.pt"),
                "--dynamic-dir",
                str(data_root / "era5_dynamic"),
                "--target-dir",
                str(data_root / "targets"),
                "--checkpoint-dir",
                str(run_dir),
            ]
            if train_months:
                train_cmd.extend(["--train-months", *train_months])
            if val_months:
                train_cmd.extend(["--val-months", *val_months])
            if resume and (run_dir / "last.pt").exists():
                train_cmd.append("--resume")
            run(train_cmd, args.dry_run)

            with status_path.open("w", encoding="utf-8") as handle:
                json.dump({"status": "completed", "run_name": run_name}, handle, indent=2)
        except Exception as exc:
            with status_path.open("w", encoding="utf-8") as handle:
                json.dump({"status": "failed", "run_name": run_name, "error": str(exc)}, handle, indent=2)
            if not continue_on_error:
                raise

    if not args.dry_run:
        write_summary(runs_root, selection_metric)


if __name__ == "__main__":
    main()
