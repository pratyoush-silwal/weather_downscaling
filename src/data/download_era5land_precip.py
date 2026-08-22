#!/usr/bin/env python3
"""Download ERA5-Land precipitation only, separate from the main target files."""

from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import cdsapi
import yaml


DATASET = "reanalysis-era5-land"
DEFAULT_TIMES = [f"{hour:02d}:00" for hour in range(24)]
VARIABLES = ["total_precipitation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5-Land precipitation from CDS.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--format", choices=["netcdf", "grib"], default=None)
    parser.add_argument("--times", nargs="+", default=None)
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


def parse_date(raw) -> date:
    return datetime.strptime(str(raw), "%Y-%m-%d").date()


def month_chunks(start: date, end: date) -> Iterable[tuple[int, int, list[str]]]:
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    while current <= final:
        _, days_in_month = calendar.monthrange(current.year, current.month)
        first_day = start.day if (current.year, current.month) == (start.year, start.month) else 1
        last_day = end.day if (current.year, current.month) == (end.year, end.month) else days_in_month
        yield current.year, current.month, [f"{day:02d}" for day in range(first_day, last_day + 1)]
        current = date(current.year + (current.month == 12), (current.month % 12) + 1, 1)


def area_from_region(region: dict) -> list[float]:
    return [
        float(region["lat_max"]),
        float(region["lon_min"]),
        float(region["lat_min"]),
        float(region["lon_max"]),
    ]


def retrieve(client: cdsapi.Client, request: dict, target: Path, overwrite: bool) -> None:
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        print(f"Skipping existing {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} precipitation -> {target}")
    client.retrieve(DATASET, request, str(target))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cfg = config["era5land"]
    start = parse_date(args.start_date or cfg["start_date"])
    end = parse_date(args.end_date or cfg["end_date"])
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    output_dir = resolve_path(args.output_dir or cfg["precipitation_output_dir"])
    data_format = args.format or cfg.get("format", "netcdf")
    times = list(args.times or cfg.get("times", DEFAULT_TIMES))
    area = area_from_region(config["region"])
    client = cdsapi.Client()

    for year, month, days in month_chunks(start, end):
        suffix = "nc" if data_format == "netcdf" else "grib"
        target = output_dir / f"era5land_precip_{year}{month:02d}.{suffix}"
        request = {
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": days,
            "time": times,
            "area": area,
            "data_format": data_format,
            "download_format": "unarchived",
        }
        retrieve(client, request, target, args.overwrite)


if __name__ == "__main__":
    main()
