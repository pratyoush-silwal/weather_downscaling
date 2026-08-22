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


TARGET_NAMES = ["t2m_target", "precipitation_target", "u10m_target", "v10m_target"]
VARIABLE_CANDIDATES = {
    "t2m": ["t2m", "2t", "2m_temperature"],
    "tp": ["tp", "total_precipitation"],
    "u10": ["u10", "10u", "10m_u_component_of_wind"],
    "v10": ["v10", "10v", "10m_v_component_of_wind"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ERA5-Land targets onto graph nodes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--precip-input-dir", default=None)
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


def _timestamps(ds) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(ds[time_name(ds)].values)).sort_values()


def _load_interp(xr, ds, logical_name: str, lat: np.ndarray, lon: np.ndarray, method: str):
    timestamps = _timestamps(ds)
    data = interpolate_to_nodes(xr, select_times(ds[variable_name(ds, logical_name)], timestamps), lat, lon, method).values
    return timestamps, data


def preprocess_month(
    xr,
    graph: dict,
    input_dir: Path,
    precip_input_dir: Path,
    output_dir: Path,
    yyyymm: str,
    method: str,
    overwrite: bool,
) -> bool:
    input_path = input_dir / f"era5land_{yyyymm}.nc"
    precip_path = precip_input_dir / f"era5land_precip_{yyyymm}.nc"
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
        timestamps, t2m = _load_interp(xr, ds, "t2m", lat, lon, method)
        _, u10 = _load_interp(xr, ds, "u10", lat, lon, method)
        _, v10 = _load_interp(xr, ds, "v10", lat, lon, method)
        if any(name in ds.data_vars for name in VARIABLE_CANDIDATES["tp"]):
            precip_times, tp = _load_interp(xr, ds, "tp", lat, lon, method)
        elif precip_path.exists():
            with open_dataset(xr, precip_path) as precip_ds:
                precip_times, tp = _load_interp(xr, precip_ds, "tp", lat, lon, method)
        else:
            raise FileNotFoundError(f"Missing precipitation for {yyyymm}: neither {input_path} nor {precip_path} has it")

    common = timestamps.intersection(precip_times)
    if common.empty:
        raise ValueError(f"No common timestamps for {yyyymm} between {input_path} and precipitation data")
    time_index = pd.Index(timestamps)
    precip_index = pd.Index(precip_times)
    base_idx = time_index.get_indexer(common)
    tp_idx = precip_index.get_indexer(common)

    y = np.stack(
        [t2m[base_idx], tp[tp_idx], u10[base_idx], v10[base_idx]],
        axis=-1,
    ).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    torch.save(
        {
            "y": torch.from_numpy(y),
            "timestamps": [str(value) for value in common],
            "target_names": TARGET_NAMES,
            "metadata": {
                "month": yyyymm,
                "source_file": str(input_path),
                "precipitation_file": str(precip_path if precip_path.exists() else input_path),
                "interpolation": method,
            },
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
    precip_input_dir = resolve_path(args.precip_input_dir or cfg["precipitation_output_dir"])
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
        written += int(
            preprocess_month(
                xr,
                graph,
                input_dir,
                precip_input_dir,
                output_dir,
                yyyymm,
                args.method,
                args.overwrite,
            )
        )
    print(f"Processed {written} month(s)")


if __name__ == "__main__":
    main()
