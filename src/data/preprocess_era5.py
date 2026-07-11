#!/usr/bin/env python3
"""Convert raw monthly ERA5 NetCDF files into node-aligned dynamic tensors.

For every complete month, this script interpolates ERA5 fields onto the static
graph nodes and writes:

    x_dynamic:     float32 [T, N, 5]
    time_features: float32 [T, 2]
    timestamps:    list[str] length T

The static graph stores topology and terrain once. Dataset code combines
``x_dynamic[t] + static_x + time_features[t]`` at training time.
"""

from __future__ import annotations

import argparse
import calendar
import math
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ATMOSPHERIC_CHANNELS = [
    "t2m_coarse",
    "q850_coarse",
    "u10m_coarse",
    "v10m_coarse",
    "z500_coarse",
]

OUTPUT_FEATURE_NAMES = [
    *ATMOSPHERIC_CHANNELS,
    "elevation_m",
    "slope_rad",
    "aspect_rad",
    "day_of_year_sin",
    "day_of_year_cos",
]

TEMPORAL_FEATURE_NAMES = [
    "day_of_year_sin",
    "day_of_year_cos",
]

VARIABLE_CANDIDATES = {
    "t2m": ["t2m", "2t", "2m_temperature"],
    "u10": ["u10", "10u", "10m_u_component_of_wind"],
    "v10": ["v10", "10v", "10m_v_component_of_wind"],
    "q850": ["q", "specific_humidity"],
    "z500": ["z", "geopotential"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ERA5 monthly files onto graph nodes.")
    parser.add_argument("--config", default="configs/default.yaml", help="Project YAML config.")
    parser.add_argument("--graph", default=None, help="Input graph package. Defaults to paths.graph_output.")
    parser.add_argument("--input-dir", default=None, help="Raw ERA5 directory. Defaults to era5.output_dir.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/era5_dynamic",
        help="Directory for processed monthly dynamic tensors.",
    )
    parser.add_argument("--start-date", default=None, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Optional explicit months as YYYYMM values, e.g. --months 199001 199002.",
    )
    parser.add_argument(
        "--method",
        choices=["linear", "nearest"],
        default="linear",
        help="Spatial interpolation method used by xarray.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Process a month even if one or more of the three raw files is missing.",
    )
    parser.add_argument(
        "--z500-as-height",
        action="store_true",
        help="Convert 500 hPa geopotential from m^2 s^-2 to geopotential height in meters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed monthly tensors.",
    )
    return parser.parse_args()


def require_xarray():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "preprocess_era5.py needs xarray plus a NetCDF4 backend. Install one of:\n"
            "  pip install xarray netCDF4\n"
            "or:\n"
            "  pip install xarray h5netcdf\n"
        ) from exc
    return xr


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_date(raw) -> date:
    if isinstance(raw, date):
        return raw
    return datetime.strptime(str(raw), "%Y-%m-%d").date()


def month_range(start: date, end: date) -> list[str]:
    months: list[str] = []
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    while current <= final:
        months.append(f"{current.year}{current.month:02d}")
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def load_graph(path: Path) -> dict:
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        graph = torch.load(path, map_location="cpu")
    for key in ["pos", "edge_index", "edge_attr"]:
        if key not in graph:
            raise ValueError(f"{path} is missing required key: {key}")
    return graph


def raw_paths(input_dir: Path, yyyymm: str) -> dict[str, Path]:
    return {
        "single": input_dir / f"era5_single_levels_{yyyymm}.nc",
        "q850": input_dir / f"era5_specific_humidity_850hpa_{yyyymm}.nc",
        "z500": input_dir / f"era5_geopotential_500hpa_{yyyymm}.nc",
    }


def open_dataset(xr, path: Path):
    # Let xarray choose netcdf4/h5netcdf/scipy based on installed backends.
    return xr.open_dataset(path)


def coord_name(ds, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.variables or name in ds.dims:
            return name
    raise ValueError(f"Could not find coordinate from candidates {candidates}; available={list(ds.variables)}")


def time_name(ds) -> str:
    return coord_name(ds, ["valid_time", "time"])


def lat_name(ds) -> str:
    return coord_name(ds, ["latitude", "lat"])


def lon_name(ds) -> str:
    return coord_name(ds, ["longitude", "lon"])


def variable_name(ds, logical_name: str) -> str:
    for name in VARIABLE_CANDIDATES[logical_name]:
        if name in ds.data_vars:
            return name
    raise ValueError(
        f"Could not find variable for {logical_name}; "
        f"candidates={VARIABLE_CANDIDATES[logical_name]}, available={list(ds.data_vars)}"
    )


def squeeze_non_spacetime(da, time_dim: str, lat_dim: str, lon_dim: str):
    keep = {time_dim, lat_dim, lon_dim}
    for dim in list(da.dims):
        if dim not in keep:
            if da.sizes[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} with size {da.sizes[dim]}")
            da = da.isel({dim: 0})
    return da


def normalize_longitudes(lon: np.ndarray, target: np.ndarray) -> np.ndarray:
    if float(np.nanmax(lon)) > 180.0 and float(np.nanmin(target)) < 0.0:
        return np.mod(target, 360.0)
    if float(np.nanmin(lon)) < 0.0 and float(np.nanmax(target)) > 180.0:
        return ((target + 180.0) % 360.0) - 180.0
    return target


def interpolate_to_nodes(xr, da, lat: np.ndarray, lon: np.ndarray, method: str):
    t_name = time_name(da.to_dataset(name="tmp"))
    y_name = lat_name(da.to_dataset(name="tmp"))
    x_name = lon_name(da.to_dataset(name="tmp"))
    da = squeeze_non_spacetime(da, t_name, y_name, x_name)

    source_lats = np.asarray(da[y_name].values, dtype=np.float64)
    source_lons = np.asarray(da[x_name].values, dtype=np.float64)
    target_lons = normalize_longitudes(source_lons, lon)

    if source_lats[0] > source_lats[-1]:
        da = da.sortby(y_name)
    if source_lons[0] > source_lons[-1]:
        da = da.sortby(x_name)

    node_lat = xr.DataArray(lat.astype(np.float64), dims="node")
    node_lon = xr.DataArray(target_lons.astype(np.float64), dims="node")
    out = da.interp({y_name: node_lat, x_name: node_lon}, method=method)
    if out.isnull().any():
        nearest = da.interp({y_name: node_lat, x_name: node_lon}, method="nearest")
        out = out.fillna(nearest)
    return out.transpose(t_name, "node")


def common_times(arrays: dict[str, object]) -> pd.DatetimeIndex:
    time_sets = []
    for da in arrays.values():
        t_name = time_name(da.to_dataset(name="tmp"))
        time_sets.append(pd.DatetimeIndex(pd.to_datetime(da[t_name].values)))
    common = time_sets[0]
    for values in time_sets[1:]:
        common = common.intersection(values)
    if common.empty:
        raise ValueError("No common timestamps found across ERA5 variables")
    return common.sort_values()


def select_times(da, times: pd.DatetimeIndex):
    t_name = time_name(da.to_dataset(name="tmp"))
    return da.sel({t_name: times})


def month_is_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def valid_torch_archive(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def temporal_features(times: pd.DatetimeIndex) -> np.ndarray:
    out = np.empty((len(times), 2), dtype=np.float32)
    for idx, timestamp in enumerate(times):
        day_angle = 2.0 * math.pi * int(timestamp.dayofyear) / 365.0
        out[idx, 0] = math.sin(day_angle)
        out[idx, 1] = math.cos(day_angle)
    return out


def preprocess_month(xr, graph: dict, input_dir: Path, output_dir: Path, yyyymm: str, args) -> bool:
    paths = raw_paths(input_dir, yyyymm)
    output_path = output_dir / f"era5_dynamic_{yyyymm}.pt"
    if valid_torch_archive(output_path) and not args.overwrite:
        print(f"Skipping existing {output_path}")
        return False
    if output_path.exists() and not args.overwrite:
        print(f"Rewriting invalid or partial output {output_path}")

    missing = [str(path) for path in paths.values() if not path.exists() or path.stat().st_size == 0]
    if missing and not args.allow_partial:
        print(f"Skipping {yyyymm}; missing files: {', '.join(missing)}")
        return False
    if missing:
        raise ValueError("--allow-partial is set, but partial atmospheric tensors are not supported yet")

    pos = graph["pos"].detach().cpu().numpy().astype(np.float32)
    lat = pos[:, 0]
    lon = pos[:, 1]
    node_count = pos.shape[0]

    with open_dataset(xr, paths["single"]) as single, open_dataset(xr, paths["q850"]) as q850, open_dataset(
        xr, paths["z500"]
    ) as z500:
        raw_arrays = {
            "t2m": single[variable_name(single, "t2m")],
            "u10": single[variable_name(single, "u10")],
            "v10": single[variable_name(single, "v10")],
            "q850": q850[variable_name(q850, "q850")],
            "z500": z500[variable_name(z500, "z500")],
        }
        times = common_times(raw_arrays)
        arrays = {
            name: interpolate_to_nodes(xr, select_times(da, times), lat, lon, args.method)
            for name, da in raw_arrays.items()
        }
        x_dynamic = np.stack(
            [
                arrays["t2m"].values,
                arrays["q850"].values,
                arrays["u10"].values,
                arrays["v10"].values,
                arrays["z500"].values / 9.80665 if args.z500_as_height else arrays["z500"].values,
            ],
            axis=-1,
        ).astype(np.float32)

    time_features = temporal_features(times)
    if not np.isfinite(x_dynamic).all():
        bad = int((~np.isfinite(x_dynamic)).sum())
        raise ValueError(f"{yyyymm} produced {bad} non-finite feature values")

    package = {
        "x_dynamic": torch.from_numpy(x_dynamic),
        "time_features": torch.from_numpy(time_features),
        "timestamps": [str(value) for value in times],
        "dynamic_feature_names": ATMOSPHERIC_CHANNELS,
        "temporal_feature_names": TEMPORAL_FEATURE_NAMES,
        "node_count": int(node_count),
        "metadata": {
            "source_files": {key: str(path) for key, path in paths.items()},
            "month": yyyymm,
            "interpolation": args.method,
            "z500_units": "m" if args.z500_as_height else "m^2 s^-2",
            "static_graph_format_expected": "static_terrain_graph_v2",
            "full_feature_names_when_combined": OUTPUT_FEATURE_NAMES,
            "format": "era5_monthly_dynamic_tensor_v2",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    torch.save(package, tmp_path)
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} x_dynamic={tuple(x_dynamic.shape)}")
    return True


def main() -> None:
    args = parse_args()
    xr = require_xarray()
    config = load_config(args.config)
    era5_config = config.get("era5", {})

    graph_path = resolve_path(args.graph or config["paths"]["graph_output"])
    input_dir = resolve_path(args.input_dir or era5_config.get("output_dir", "data/raw/era5"))
    output_dir = resolve_path(args.output_dir)

    if args.months:
        months = args.months
    else:
        start = parse_date(args.start_date or era5_config.get("start_date"))
        end = parse_date(args.end_date or era5_config.get("end_date"))
        months = month_range(start, end)

    graph = load_graph(graph_path)
    written = 0
    for yyyymm in months:
        if len(yyyymm) != 6 or not yyyymm.isdigit():
            raise ValueError(f"Invalid month {yyyymm}; expected YYYYMM")
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month {yyyymm}")
        calendar.monthrange(year, month)
        written += int(preprocess_month(xr, graph, input_dir, output_dir, yyyymm, args))

    print(f"Processed {written} month(s)")


if __name__ == "__main__":
    main()
