#!/usr/bin/env python3
"""Build the static Nepal terrain-aware graph described in the project report.

The saved package is a plain ``torch.save`` dictionary so it works without
torch_geometric. The topology and terrain features are static and should be
reused for every ERA5 timestep:

    pos:        float32 [N, 2]
    edge_index: int64   [2, E]
    edge_attr:  float32 [E, 3]
    static_x:   float32 [N, 3], elevation/slope/aspect

For backward compatibility the package also includes an ``x``/``x_raw``
template with zero atmospheric channels and placeholder temporal channels.
Training should use ``static_x`` plus dynamic ERA5 tensors produced by
``preprocess_era5.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree


NODE_FEATURE_NAMES = [
    "t2m_coarse",
    "q850_coarse",
    "u10m_coarse",
    "v10m_coarse",
    "z500_coarse",
    "tp_coarse",
    "elevation_m",
    "slope_rad",
    "aspect_rad",
    "day_of_year_sin",
    "day_of_year_cos",
]

DYNAMIC_FEATURE_NAMES = [
    "t2m_coarse",
    "q850_coarse",
    "u10m_coarse",
    "v10m_coarse",
    "z500_coarse",
    "tp_coarse",
]

STATIC_FEATURE_NAMES = [
    "elevation_m",
    "slope_rad",
    "aspect_rad",
]

TEMPORAL_FEATURE_NAMES = [
    "day_of_year_sin",
    "day_of_year_cos",
]

EDGE_ATTR_NAMES = [
    "haversine_distance_m",
    "elevation_delta_m",
    "aspect_alignment",
]

TIFF_TYPES = {
    1: ("B", 1),
    2: ("c", 1),
    3: ("H", 2),
    4: ("I", 4),
    5: ("II", 8),
    11: ("f", 4),
    12: ("d", 8),
}


@dataclass(frozen=True)
class GeoTiffInfo:
    path: Path
    endian: str
    width: int
    height: int
    rows_per_strip: int
    strip_offsets: tuple[int, ...]
    strip_byte_counts: tuple[int, ...]
    lon_min: float
    lat_max: float
    pixel_width: float
    pixel_height: float
    nodata: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Nepal PI-GNN graph package."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dem", default=None, help="Override DEM path from config.")
    parser.add_argument(
        "--output", default=None, help="Override graph output path from config."
    )
    parser.add_argument(
        "--stations",
        default=None,
        help=(
            "Optional CSV of station nodes. Expected columns: lat, lon, and optionally "
            "elevation or elevation_m."
        ),
    )
    parser.add_argument(
        "--day-of-year",
        type=int,
        default=1,
        help="Day-of-year used for the temporal sin/cos placeholder channels.",
    )
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="Z-score the 10 channels using this graph snapshot statistics.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def decode_inline_value(type_id: int, count: int, value: bytes, endian: str):
    fmt, size = TIFF_TYPES[type_id]
    raw = value[: count * size]
    if type_id == 2:
        return raw.rstrip(b"\x00").decode("ascii")
    values = struct.unpack(endian + fmt * count, raw)
    return values[0] if count == 1 else values


def read_tiff_tags(path: Path) -> tuple[str, dict[int, tuple[int, int, object]]]:
    with path.open("rb") as handle:
        byte_order = handle.read(2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError(f"{path} is not a classic TIFF file")

        version = struct.unpack(endian + "H", handle.read(2))[0]
        if version != 42:
            raise ValueError("Only classic TIFF is supported")

        ifd_offset = struct.unpack(endian + "I", handle.read(4))[0]
        handle.seek(ifd_offset)
        entry_count = struct.unpack(endian + "H", handle.read(2))[0]
        tags: dict[int, tuple[int, int, object]] = {}

        for _ in range(entry_count):
            tag, type_id, count, raw_value = struct.unpack(
                endian + "HHI4s", handle.read(12)
            )
            if type_id not in TIFF_TYPES:
                continue
            fmt, size = TIFF_TYPES[type_id]
            byte_count = count * size
            if byte_count <= 4:
                value = decode_inline_value(type_id, count, raw_value, endian)
            else:
                offset = struct.unpack(endian + "I", raw_value)[0]
                current = handle.tell()
                handle.seek(offset)
                data = handle.read(byte_count)
                handle.seek(current)
                if type_id == 2:
                    value = data.rstrip(b"\x00").decode("ascii")
                else:
                    values = struct.unpack(endian + fmt * count, data)
                    value = values[0] if count == 1 else values
            tags[tag] = (type_id, count, value)

    return endian, tags


def require_tag(tags: dict[int, tuple[int, int, object]], tag: int):
    if tag not in tags:
        raise ValueError(f"Required TIFF tag {tag} is missing")
    return tags[tag][2]


def read_geotiff_info(path: Path) -> GeoTiffInfo:
    endian, tags = read_tiff_tags(path)
    bits = int(require_tag(tags, 258))
    sample_format = int(require_tag(tags, 339))
    if bits != 16 or sample_format != 2:
        raise ValueError("Expected signed int16 DEM GeoTIFF")

    strip_offsets = require_tag(tags, 273)
    strip_byte_counts = require_tag(tags, 279)
    if isinstance(strip_offsets, int):
        strip_offsets = (strip_offsets,)
    if isinstance(strip_byte_counts, int):
        strip_byte_counts = (strip_byte_counts,)

    pixel_scale = require_tag(tags, 33550)
    tiepoint = require_tag(tags, 33922)
    nodata_raw = tags.get(42113, (None, None, "-32768"))[2]

    return GeoTiffInfo(
        path=path,
        endian=endian,
        width=int(require_tag(tags, 256)),
        height=int(require_tag(tags, 257)),
        rows_per_strip=int(require_tag(tags, 278)),
        strip_offsets=tuple(int(v) for v in strip_offsets),
        strip_byte_counts=tuple(int(v) for v in strip_byte_counts),
        lon_min=float(pixel_scale and tiepoint[3]),
        lat_max=float(pixel_scale and tiepoint[4]),
        pixel_width=float(pixel_scale[0]),
        pixel_height=float(pixel_scale[1]),
        nodata=int(str(nodata_raw)),
    )


def make_regular_nodes(region: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lat_min = float(region["lat_min"])
    lat_max = float(region["lat_max"])
    lon_min = float(region["lon_min"])
    lon_max = float(region["lon_max"])
    resolution = float(region["resolution_deg"])

    lat_count = int(round((lat_max - lat_min) / resolution)) + 1
    lon_count = int(round((lon_max - lon_min) / resolution)) + 1
    lats = lat_min + np.arange(lat_count, dtype=np.float64) * resolution
    lons = lon_min + np.arange(lon_count, dtype=np.float64) * resolution
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    node_type = ["grid"] * coords.shape[0]
    return coords[:, 0], coords[:, 1], node_type


def read_station_nodes(
    path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    if path is None:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), None, []

    lats: list[float] = []
    lons: list[float] = []
    elevations: list[float] = []
    has_elevation = False
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            if lat is None or lon is None:
                raise ValueError("Station CSV needs lat/lon columns")
            lats.append(float(lat))
            lons.append(float(lon))
            elev = row.get("elevation_m") or row.get("elevation")
            if elev not in (None, ""):
                has_elevation = True
                elevations.append(float(elev))
            else:
                elevations.append(float("nan"))

    elev_array = np.asarray(elevations, dtype=np.float64) if has_elevation else None
    return (
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        elev_array,
        ["station"] * len(lats),
    )


def latlon_to_pixel(
    info: GeoTiffInfo, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cols = np.rint((lon - info.lon_min) / info.pixel_width).astype(np.int64)
    rows = np.rint((info.lat_max - lat) / info.pixel_height).astype(np.int64)
    rows = np.clip(rows, 1, info.height - 2)
    cols = np.clip(cols, 1, info.width - 2)
    return rows, cols


def read_dem_rows(info: GeoTiffInfo, rows: Iterable[int]) -> dict[int, np.ndarray]:
    wanted = sorted(set(int(r) for r in rows))
    wanted_set = set(wanted)
    out: dict[int, np.ndarray] = {}
    with info.path.open("rb") as handle:
        for strip_index, offset in enumerate(info.strip_offsets):
            first_row = strip_index * info.rows_per_strip
            row_count = min(info.rows_per_strip, info.height - first_row)
            strip_rows = [
                row for row in wanted if first_row <= row < first_row + row_count
            ]
            if not strip_rows:
                continue
            handle.seek(offset)
            raw = handle.read(info.strip_byte_counts[strip_index])
            strip = np.frombuffer(raw, dtype=info.endian + "i2").reshape(
                row_count, info.width
            )
            for row in strip_rows:
                if row in wanted_set:
                    out[row] = strip[row - first_row, :].astype(np.float32)
    missing = wanted_set - set(out)
    if missing:
        raise ValueError(f"Could not read DEM rows: {sorted(missing)[:5]}")
    return out


def sample_terrain(
    info: GeoTiffInfo,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = latlon_to_pixel(info, lat, lon)
    needed_rows = np.concatenate([rows - 1, rows, rows + 1])
    row_cache = read_dem_rows(info, needed_rows)

    elevation = np.empty(lat.shape[0], dtype=np.float32)
    slope = np.empty(lat.shape[0], dtype=np.float32)
    aspect = np.empty(lat.shape[0], dtype=np.float32)

    for idx, (row, col, lat_value) in enumerate(zip(rows, cols, lat)):
        window = np.stack(
            [
                row_cache[int(row - 1)][col - 1 : col + 2],
                row_cache[int(row)][col - 1 : col + 2],
                row_cache[int(row + 1)][col - 1 : col + 2],
            ]
        )
        if np.any(window == info.nodata):
            window = np.where(window == info.nodata, np.nan, window)

        z = window[1, 1]
        elevation[idx] = z if np.isfinite(z) else np.nanmean(window)

        dz_dcol = (
            (window[0, 2] + 2 * window[1, 2] + window[2, 2])
            - (window[0, 0] + 2 * window[1, 0] + window[2, 0])
        ) / 8.0
        dz_drow = (
            (window[2, 0] + 2 * window[2, 1] + window[2, 2])
            - (window[0, 0] + 2 * window[0, 1] + window[0, 2])
        ) / 8.0

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = meters_per_deg_lat * math.cos(
            math.radians(float(lat_value))
        )
        dz_dx = dz_dcol / (info.pixel_width * meters_per_deg_lon)
        dz_dy = -dz_drow / (info.pixel_height * meters_per_deg_lat)

        slope[idx] = math.atan(math.sqrt(float(dz_dx * dz_dx + dz_dy * dz_dy)))
        aspect_value = math.atan2(float(dz_dx), float(dz_dy))
        aspect[idx] = aspect_value if aspect_value >= 0 else aspect_value + 2 * math.pi

    return elevation, slope, aspect


def geodetic_to_ecef(
    lat: np.ndarray, lon: np.ndarray, height_m: np.ndarray
) -> np.ndarray:
    semi_major = 6_378_137.0
    eccentricity_sq = 6.69437999014e-3
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    normal = semi_major / np.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)

    x = (normal + height_m) * cos_lat * np.cos(lon_rad)
    y = (normal + height_m) * cos_lat * np.sin(lon_rad)
    z = (normal * (1.0 - eccentricity_sq) + height_m) * sin_lat
    return np.column_stack([x, y, z])


def haversine_distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    radius = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_edges(
    lat: np.ndarray,
    lon: np.ndarray,
    elevation: np.ndarray,
    aspect: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    ecef = geodetic_to_ecef(lat, lon, elevation)
    tree = cKDTree(ecef)
    _, neighbor_idx = tree.query(ecef, k=k + 1)
    neighbor_idx = neighbor_idx[:, 1:]

    src = np.repeat(np.arange(lat.shape[0], dtype=np.int64), k)
    dst = neighbor_idx.reshape(-1).astype(np.int64)
    edge_index = np.vstack([src, dst])

    distance = haversine_distance_m(lat[src], lon[src], lat[dst], lon[dst])
    elevation_delta = elevation[dst] - elevation[src]
    aspect_alignment = np.cos(aspect[src] - aspect[dst])
    edge_attr = np.column_stack([distance, elevation_delta, aspect_alignment]).astype(
        np.float32
    )
    return edge_index, edge_attr


def normalise_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((x - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    region = config["region"]
    graph_config = config["graph"]
    paths = config["paths"]

    dem_path = resolve_path(args.dem or paths["dem_raw"])
    output_path = resolve_path(args.output or paths["graph_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid_lat, grid_lon, node_types = make_regular_nodes(region)
    station_lat, station_lon, station_elev, station_types = read_station_nodes(
        args.stations
    )
    lat = np.concatenate([grid_lat, station_lat])
    lon = np.concatenate([grid_lon, station_lon])
    node_types.extend(station_types)

    info = read_geotiff_info(dem_path)
    elevation, slope, aspect = sample_terrain(info, lat, lon)
    if station_elev is not None and station_elev.size:
        start = grid_lat.shape[0]
        mask = np.isfinite(station_elev)
        elevation[start : start + station_elev.shape[0]][mask] = station_elev[
            mask
        ].astype(np.float32)

    day_angle = 2.0 * math.pi * int(args.day_of_year) / 365.0
    atmospheric = np.zeros((lat.shape[0], 6), dtype=np.float32)  # replace after era5
    temporal = np.tile(
        np.array([math.sin(day_angle), math.cos(day_angle)], dtype=np.float32),
        (lat.shape[0], 1),
    )
    terrain = np.column_stack([elevation, slope, aspect]).astype(np.float32)
    x_raw = np.column_stack([atmospheric, terrain, temporal]).astype(np.float32)

    if args.normalise:
        x, feature_mean, feature_std = normalise_features(x_raw)
    else:
        x = x_raw
        feature_mean = np.zeros(x.shape[1], dtype=np.float32)
        feature_std = np.ones(x.shape[1], dtype=np.float32)

    edge_index, edge_attr = build_edges(
        lat=lat,
        lon=lon,
        elevation=elevation,
        aspect=aspect,
        k=int(graph_config["k_neighbours"]),
    )

    y = np.full((lat.shape[0], 4), np.nan, dtype=np.float32)
    package = {
        "x": torch.from_numpy(x),
        "x_raw": torch.from_numpy(x_raw),
        "static_x": torch.from_numpy(terrain),
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(edge_attr),
        "y": torch.from_numpy(y),
        "pos": torch.from_numpy(np.column_stack([lat, lon]).astype(np.float32)),
        "elevation": torch.from_numpy(elevation.astype(np.float32)),
        "slope": torch.from_numpy(slope.astype(np.float32)),
        "aspect": torch.from_numpy(aspect.astype(np.float32)),
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "metadata": {
            "region": region,
            "node_feature_names": NODE_FEATURE_NAMES,
            "dynamic_feature_names": DYNAMIC_FEATURE_NAMES,
            "static_feature_names": STATIC_FEATURE_NAMES,
            "temporal_feature_names": TEMPORAL_FEATURE_NAMES,
            "edge_attr_names": EDGE_ATTR_NAMES,
            "label_names": ["t2m_target", "precipitation_target", "u10m_target", "v10m_target"],
            "node_type_counts": {
                "grid": node_types.count("grid"),
                "station": node_types.count("station"),
            },
            "node_types": node_types,
            "dem_path": str(dem_path),
            "station_path": str(resolve_path(args.stations)) if args.stations else None,
            "day_of_year": int(args.day_of_year),
            "normalised": bool(args.normalise),
            "format": "static_terrain_graph_v2",
            "compatibility_note": (
                "x/x_raw are templates only; dynamic ERA5 atmospheric and temporal "
                "features are stored separately and combined by dataset.py."
            ),
        },
    }

    torch.save(package, output_path)
    print(f"Wrote {output_path}")
    print(f"nodes: {lat.shape[0]}")
    print(f"edges: {edge_index.shape[1]}")
    print(f"x: {tuple(package['x'].shape)}")
    print(f"edge_index: {tuple(package['edge_index'].shape)}")
    print(f"edge_attr: {tuple(package['edge_attr'].shape)}")
    print(
        f"elevation range: {float(np.nanmin(elevation)):.1f} m to {float(np.nanmax(elevation)):.1f} m"
    )


if __name__ == "__main__":
    main()
