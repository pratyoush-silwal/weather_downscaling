#!/usr/bin/env python3
"""Repack preprocessed ERA5-Land targets into final monthly tensors."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import torch
import yaml


FINAL_TARGET_NAMES = [
    "t2m_target",
    "precipitation_target",
    "u10m_target",
    "v10m_target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final monthly target tensors.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--era5land-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def valid_torch_archive(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def month_from_path(path: Path) -> str | None:
    match = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])", path.stem)
    return match.group(0) if match else None


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_month(era5land_path: Path, output_path: Path, overwrite: bool) -> bool:
    if valid_torch_archive(output_path) and not overwrite:
        print(f"Skipping existing {output_path}")
        return False
    era5land = _load_torch(era5land_path)
    y = era5land["y"].float()
    if y.shape[-1] != 4:
        raise ValueError(f"{era5land_path} expected 4 ERA5-Land target channels, got {y.shape[-1]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    torch.save(
        {
            "y": y,
            "timestamps": list(era5land["timestamps"]),
            "target_names": FINAL_TARGET_NAMES,
            "metadata": {
                "era5land_file": str(era5land_path),
                "month": month_from_path(output_path),
            },
        },
        tmp_path,
    )
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} y={tuple(y.shape)}")
    return True


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    era5land_dir = resolve_path(args.era5land_dir or config["era5land"]["processed_output_dir"])
    output_dir = resolve_path(args.output_dir or config["targets"]["output_dir"])

    era5land_files = {month_from_path(path): path for path in sorted(era5land_dir.glob("*.pt")) if month_from_path(path)}
    months = sorted(era5land_files)

    written = 0
    for month in months:
        output_path = output_dir / f"targets_{month}.pt"
        written += int(build_month(era5land_files[month], output_path, args.overwrite))
    print(f"Processed {written} month(s)")


if __name__ == "__main__":
    main()
