#!/usr/bin/env python3
"""Run the baseline/GNN jobs from the notebook and write a Markdown run log."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline and GNN jobs without Jupyter.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report", default="notebooks/cloud/run_weather_baselines_report.md")
    parser.add_argument("--smoke", action="store_true", help="Use one month for train/validation.")
    parser.add_argument("--smoke-month", default="199001")
    parser.add_argument("--train-months", nargs="*", default=None)
    parser.add_argument("--val-months", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-interpolation", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-gnn", action="store_true")
    return parser.parse_args()


def build_month_args(train_months: list[str] | None, val_months: list[str] | None) -> list[str]:
    args: list[str] = []
    if train_months:
        args.extend(["--train-months", *train_months])
    if val_months:
        args.extend(["--val-months", *val_months])
    return args


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_command(name: str, command: list[str]) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=repo_root(),
        text=True,
        capture_output=True,
    )
    return CommandResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def code_block(text: str, language: str = "") -> str:
    fence = f"```{language}".rstrip()
    return f"{fence}\n{text}\n```"


def format_json(data: dict[str, Any] | None) -> str:
    return "not found" if data is None else json.dumps(data, indent=2)


def write_report(
    path: Path,
    config_path: Path,
    config: dict[str, Any],
    train_months: list[str] | None,
    val_months: list[str] | None,
    results: list[CommandResult],
) -> None:
    baseline_root = resolve_path(config["baselines"]["output_dir"])
    checkpoint_root = resolve_path(config["training"]["checkpoint_dir"])
    artifacts = {
        "interpolation": read_json_if_exists(baseline_root / "interpolation" / "metrics.json"),
        "mlp": read_json_if_exists(baseline_root / "mlp" / "metrics.json"),
        "xgboost": read_json_if_exists(baseline_root / "xgboost" / "metrics.json"),
        "gnn": read_json_if_exists(checkpoint_root / "metrics.json"),
    }

    lines = [
        "# Weather Training Run Log",
        "",
        "## Configuration",
        "",
        f"- config: `{config_path}`",
        f"- device: `{config['training']['device']}`",
        f"- train_years: `{config['training']['train_years']}`",
        f"- validation_years: `{config['training']['validation_years']}`",
        f"- test_years: `{config['training']['test_years']}`",
        f"- custom train months: `{train_months}`",
        f"- custom validation months: `{val_months}`",
        "",
        "## Command Status",
        "",
        "| job | status | command |",
        "|---|---|---|",
    ]
    for result in results:
        status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
        lines.append(f"| {result.name} | {status} | `{shlex.join(result.command)}` |")

    for result in results:
        lines.extend(
            [
                "",
                f"## {result.name}",
                "",
                code_block(shlex.join(result.command), "bash"),
                "",
                "### stdout",
                "",
                code_block(result.stdout or "(empty)"),
            ]
        )
        if result.stderr:
            lines.extend(["", "### stderr", "", code_block(result.stderr)])

    lines.extend(["", "## Saved Metrics", ""])
    for name, metrics in artifacts.items():
        lines.extend([f"### {name}", "", code_block(format_json(metrics), "json"), ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = args.device or config["training"]["device"]
    train_months = args.train_months
    val_months = args.val_months
    if args.smoke:
        train_months = [args.smoke_month]
        val_months = [args.smoke_month]

    month_args = build_month_args(train_months, val_months)
    results: list[CommandResult] = []

    if not args.skip_interpolation:
        results.append(
            run_command(
                "interpolation",
                ["python", "src/training/train_baseline.py", "--config", args.config, "--model", "interpolation", *month_args],
            )
        )
    if not args.skip_mlp:
        results.append(
            run_command(
                "mlp",
                ["python", "src/training/train_baseline.py", "--config", args.config, "--model", "mlp", "--device", device, *month_args],
            )
        )
    if not args.skip_xgboost:
        results.append(
            run_command(
                "xgboost",
                ["python", "src/training/train_baseline.py", "--config", args.config, "--model", "xgboost", *month_args],
            )
        )
    if not args.skip_gnn:
        results.append(
            run_command(
                "gnn",
                ["python", "src/training/train.py", "--config", args.config, "--device", device, *month_args],
            )
        )

    write_report(resolve_path(args.report), resolve_path(args.config), config, train_months, val_months, results)

    failed = [result for result in results if result.returncode != 0]
    if failed:
        raise SystemExit(f"{len(failed)} command(s) failed; see {resolve_path(args.report)}")
    print(resolve_path(args.report))


if __name__ == "__main__":
    main()
