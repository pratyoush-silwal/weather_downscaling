#!/usr/bin/env python3
"""Preprocess MSWEP precipitation files onto graph nodes."""

from __future__ import annotations

import argparse
import calendar
import re
import zipfile
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess_era5 import (
    interpolate_to_nodes,
    load_graph,
    open_dataset,
    require_xarray,
    resolve_path,
    select_times,
    time_name,
)


TARGET_NAME = ["precipitation_target"]
VARIABLE_CANDIDATES = [
    "precipitation",
    "precip",
    "precipitation_amount",
    "precipitation_rate",
    "rain",
    "rainfall",
    "tp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MSWEP precipitation onto graph nodes.")
    parser.add_argument("--graph", default="data/processed/nepal_graph.pt")
    parser.add_argument("--input-dir", default="data/raw/mswep")
    parser.add_argument("--output-dir", default="data/processed/targets_mswep")
    parser.add_argument("--method", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_torch_archive(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def month_from_path(path: Path) -> str | None:
    match = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])", path.stem)
    return match.group(0) if match else None


def variable_name(ds) -> str:
    for name in VARIABLE_CANDIDATES:
        if name in ds.data_vars:
            return name
    raise ValueError(f"Could not find precipitation variable; available={list(ds.data_vars)}")


def preprocess_file(xr, graph: dict, input_path: Path, output_dir: Path, method: str, overwrite: bool) -> bool:
    yyyymm = month_from_path(input_path)
    if yyyymm is None:
        print(f"Skipping {input_path}; could not infer YYYYMM from filename")
        return False
    output_path = output_dir / f"mswep_precip_{yyyymm}.pt"
    if valid_torch_archive(output_path) and not overwrite:
        print(f"Skipping existing {output_path}")
        return False

    pos = graph["pos"].detach().cpu().numpy().astype("float32")
    lat = pos[:, 0]
    lon = pos[:, 1]
    with open_dataset(xr, input_path) as ds:
        precip = ds[variable_name(ds)]
        timestamps = pd.DatetimeIndex(pd.to_datetime(ds[time_name(ds)].values)).sort_values()
        values = interpolate_to_nodes(xr, select_times(precip, timestamps), lat, lon, method).values.astype("float32")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    torch.save(
        {
            "y": torch.from_numpy(values[..., None]),
            "timestamps": [str(value) for value in timestamps],
            "target_names": TARGET_NAME,
            "metadata": {"month": yyyymm, "source_file": str(input_path), "interpolation": method},
        },
        tmp_path,
    )
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} y={tuple(values.shape)}")
    return True


def main() -> None:
    args = parse_args()
    xr = require_xarray()
    graph = load_graph(resolve_path(args.graph))
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    written = 0
    for path in sorted(input_dir.glob("*.nc")):
        month = month_from_path(path)
        if month is not None:
            year = int(month[:4])
            mon = int(month[4:])
            calendar.monthrange(year, mon)
        written += int(preprocess_file(xr, graph, path, output_dir, args.method, args.overwrite))
    print(f"Processed {written} file(s)")


if __name__ == "__main__":
    main()
