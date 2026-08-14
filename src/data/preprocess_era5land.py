#!/usr/bin/env python3
"""Preprocess ERA5-Land monthly targets onto graph nodes."""

from __future__ import annotations

import argparse
import calendar
import zipfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess_era5 import (
    interpolate_to_nodes,
    load_config,
    load_graph,
    month_range,
    open_dataset,
    parse_date,
    require_xarray,
    resolve_path,
    select_times,
    time_name,
)


TARGET_NAMES = ["t2m_target", "u10m_target", "v10m_target"]
VARIABLE_CANDIDATES = {
    "t2m": ["t2m", "2t", "2m_temperature"],
    "u10": ["u10", "10u", "10m_u_component_of_wind"],
    "v10": ["v10", "10v", "10m_v_component_of_wind"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ERA5-Land targets onto graph nodes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--months", nargs="*", default=None)
    parser.add_argument("--method", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_torch_archive(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def variable_name(ds, logical_name: str) -> str:
    for name in VARIABLE_CANDIDATES[logical_name]:
        if name in ds.data_vars:
            return name
    raise ValueError(f"Missing ERA5-Land variable {logical_name}; available={list(ds.data_vars)}")


def preprocess_month(xr, graph: dict, input_dir: Path, output_dir: Path, yyyymm: str, method: str, overwrite: bool) -> bool:
    input_path = input_dir / f"era5land_{yyyymm}.nc"
    output_path = output_dir / f"era5land_targets_{yyyymm}.pt"
    if valid_torch_archive(output_path) and not overwrite:
        print(f"Skipping existing {output_path}")
        return False
    if not input_path.exists():
        print(f"Skipping {yyyymm}; missing {input_path}")
        return False

    pos = graph["pos"].detach().cpu().numpy().astype(np.float32)
    lat = pos[:, 0]
    lon = pos[:, 1]
    with open_dataset(xr, input_path) as ds:
        arrays = {
            "t2m": ds[variable_name(ds, "t2m")],
            "u10": ds[variable_name(ds, "u10")],
            "v10": ds[variable_name(ds, "v10")],
        }
        timestamps = pd.DatetimeIndex(pd.to_datetime(ds[time_name(ds)].values)).sort_values()
        interpolated = {
            name: interpolate_to_nodes(xr, select_times(da, timestamps), lat, lon, method).values
            for name, da in arrays.items()
        }

    y = np.stack(
        [interpolated["t2m"], interpolated["u10"], interpolated["v10"]],
        axis=-1,
    ).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    torch.save(
        {
            "y": torch.from_numpy(y),
            "timestamps": [str(value) for value in timestamps],
            "target_names": TARGET_NAMES,
            "metadata": {"month": yyyymm, "source_file": str(input_path), "interpolation": method},
        },
        tmp_path,
    )
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} y={tuple(y.shape)}")
    return True


def main() -> None:
    args = parse_args()
    xr = require_xarray()
    config = load_config(args.config)
    graph = load_graph(resolve_path(args.graph or config["paths"]["graph_output"]))
    cfg = config["era5land"]
    input_dir = resolve_path(args.input_dir or cfg["output_dir"])
    output_dir = resolve_path(args.output_dir or cfg["processed_output_dir"])
    months = args.months or month_range(
        parse_date(args.start_date or cfg["start_date"]),
        parse_date(args.end_date or cfg["end_date"]),
    )

    written = 0
    for yyyymm in months:
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])
        calendar.monthrange(year, month)
        written += int(preprocess_month(xr, graph, input_dir, output_dir, yyyymm, args.method, args.overwrite))
    print(f"Processed {written} month(s)")


if __name__ == "__main__":
    main()
