#!/usr/bin/env python3
"""Prepare and optionally train one GNN experiment for a specific boundary halo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph/data for one clip-buffer setting and optionally train.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--clip-buffer-deg", type=float, required=True)
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument("--months", nargs="*", default=None, help="Optional YYYYMM subset for preprocessing.")
    parser.add_argument("--train", action="store_true", help="Run src/training/train.py after preprocessing.")
    parser.add_argument("--overwrite", action="store_true")
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


def buffer_tag(value: float) -> str:
    return f"buffer_{value:.4f}".replace("-", "m").replace(".", "p")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root())


def maybe_extend(cmd: list[str], flag: str, values: list[str] | None) -> None:
    if values:
        cmd.extend([flag, *values])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tag = args.experiment_tag or buffer_tag(args.clip_buffer_deg)
    root = resolve_path(config["training"].get("experiments_root", "data/processed/buffer_experiments")) / tag
    graph_path = root / "graph.pt"
    dynamic_dir = root / "era5_dynamic"
    era5land_dir = root / "targets_era5land"
    target_dir = root / "targets"
    checkpoint_dir = resolve_path(args.checkpoint_dir or (resolve_path(config["training"]["checkpoint_dir"]) / tag))

    if args.overwrite or not graph_path.exists():
        build_graph_cmd = [
            sys.executable,
            "-m",
            "src.data.build_graph",
            "--config",
            args.config,
            "--clip-buffer-deg",
            str(args.clip_buffer_deg),
            "--output",
            str(graph_path),
        ]
        run(build_graph_cmd)
    else:
        print(f"Skipping existing {graph_path}")

    preprocess_era5_cmd = [
        sys.executable,
        "src/data/preprocess_era5.py",
        "--config",
        args.config,
        "--graph",
        str(graph_path),
        "--output-dir",
        str(dynamic_dir),
    ]
    maybe_extend(preprocess_era5_cmd, "--months", args.months)
    if args.overwrite:
        preprocess_era5_cmd.append("--overwrite")
    run(preprocess_era5_cmd)

    preprocess_era5land_cmd = [
        sys.executable,
        "src/data/preprocess_era5land.py",
        "--config",
        args.config,
        "--graph",
        str(graph_path),
        "--output-dir",
        str(era5land_dir),
    ]
    maybe_extend(preprocess_era5land_cmd, "--months", args.months)
    if args.overwrite:
        preprocess_era5land_cmd.append("--overwrite")
    run(preprocess_era5land_cmd)

    build_targets_cmd = [
        sys.executable,
        "src/data/build_targets.py",
        "--config",
        args.config,
        "--era5land-dir",
        str(era5land_dir),
        "--output-dir",
        str(target_dir),
    ]
    if args.overwrite:
        build_targets_cmd.append("--overwrite")
    run(build_targets_cmd)

    if not args.train:
        print(f"prepared {tag}")
        print(f"graph={graph_path}")
        print(f"dynamic_dir={dynamic_dir}")
        print(f"target_dir={target_dir}")
        return

    train_cmd = [
        sys.executable,
        "src/training/train.py",
        "--config",
        args.config,
        "--graph",
        str(graph_path),
        "--dynamic-dir",
        str(dynamic_dir),
        "--target-dir",
        str(target_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    maybe_extend(train_cmd, "--train-months", args.train_months)
    maybe_extend(train_cmd, "--val-months", args.val_months)
    if args.epochs is not None:
        train_cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate is not None:
        train_cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.device is not None:
        train_cmd.extend(["--device", str(args.device)])
    if args.resume:
        train_cmd.append("--resume")
    run(train_cmd)


if __name__ == "__main__":
    main()
