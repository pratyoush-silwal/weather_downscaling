#!/usr/bin/env python3
"""Download ERA5 inputs for the configured project region.

The project needs five ERA5 predictors:

* 2m temperature
* 10m u wind
* 10m v wind
* specific humidity at 850 hPa
* geopotential at 500 hPa

CDS stores single-level and pressure-level variables in separate datasets, so
this script writes three files per month by default.
"""

from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import cdsapi
import yaml


SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"
PRESSURE_LEVEL_DATASET = "reanalysis-era5-pressure-levels"

SINGLE_LEVEL_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

PRESSURE_LEVEL_REQUESTS = [
    ("specific_humidity_850hpa", "specific_humidity", "850"),
    ("geopotential_500hpa", "geopotential", "500"),
]

DEFAULT_TIMES = [f"{hour:02d}:00" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 predictors from CDS.")
    parser.add_argument("--config", default="configs/default.yaml", help="Project YAML config.")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for downloaded monthly ERA5 files.",
    )
    parser.add_argument(
        "--format",
        choices=["netcdf", "grib"],
        default=None,
        help="CDS output data format.",
    )
    parser.add_argument(
        "--times",
        nargs="+",
        default=None,
        help="UTC synoptic times, e.g. --times 00:00 06:00 12:00 18:00.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download even if the target file already exists.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def month_chunks(start: date, end: date) -> Iterable[tuple[int, int, list[str]]]:
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)

    while current <= final:
        _, days_in_month = calendar.monthrange(current.year, current.month)
        first_day = start.day if (current.year, current.month) == (start.year, start.month) else 1
        last_day = end.day if (current.year, current.month) == (end.year, end.month) else days_in_month
        days = [f"{day:02d}" for day in range(first_day, last_day + 1)]
        yield current.year, current.month, days

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def era5_area(region: dict) -> list[float]:
    # CDS area order is North, West, South, East.
    return [
        float(region["lat_max"]),
        float(region["lon_min"]),
        float(region["lat_min"]),
        float(region["lon_max"]),
    ]


def data_format_fields(data_format: str) -> dict[str, str]:
    # The current CDS API accepts data_format/download_format for ERA5.
    return {
        "data_format": data_format,
        "download_format": "unarchived",
    }


def retrieve(client: cdsapi.Client, dataset: str, request: dict, target: Path, overwrite: bool) -> None:
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        print(f"Skipping existing {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset} -> {target}")
    client.retrieve(dataset, request, str(target))


def monthly_base_request(
    *,
    year: int,
    month: int,
    days: list[str],
    times: list[str],
    area: list[float],
    data_format: str,
) -> dict:
    return {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": times,
        "area": area,
        **data_format_fields(data_format),
    }


def download_month(
    client: cdsapi.Client,
    output_dir: Path,
    year: int,
    month: int,
    days: list[str],
    times: list[str],
    area: list[float],
    data_format: str,
    overwrite: bool,
) -> None:
    suffix = "nc" if data_format == "netcdf" else "grib"
    base_request = monthly_base_request(
        year=year,
        month=month,
        days=days,
        times=times,
        area=area,
        data_format=data_format,
    )

    single_target = output_dir / f"era5_single_levels_{year}{month:02d}.{suffix}"
    retrieve(
        client,
        SINGLE_LEVEL_DATASET,
        {**base_request, "variable": SINGLE_LEVEL_VARIABLES},
        single_target,
        overwrite,
    )

    for label, variable, pressure_level in PRESSURE_LEVEL_REQUESTS:
        target = output_dir / f"era5_{label}_{year}{month:02d}.{suffix}"
        retrieve(
            client,
            PRESSURE_LEVEL_DATASET,
            {
                **base_request,
                "variable": [variable],
                "pressure_level": [pressure_level],
            },
            target,
            overwrite,
        )


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    era5_config = config.get("era5", {})

    start_date = args.start_date or era5_config.get("start_date")
    end_date = args.end_date or era5_config.get("end_date")
    if not start_date or not end_date:
        raise ValueError("Provide --start-date/--end-date or set era5.start_date/end_date in config")

    start = parse_date(str(start_date))
    end = parse_date(str(end_date))
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    region = config["region"]
    area = era5_area(region)
    output_dir = resolve_path(args.output_dir or era5_config.get("output_dir", "data/raw/era5"))
    data_format = args.format or era5_config.get("format", "netcdf")
    times = list(args.times or era5_config.get("times", DEFAULT_TIMES))
    client = cdsapi.Client()

    print(
        "ERA5 area north/west/south/east: "
        f"{area[0]}, {area[1]}, {area[2]}, {area[3]}"
    )
    for year, month, days in month_chunks(start, end):
        download_month(
            client=client,
            output_dir=output_dir,
            year=year,
            month=month,
            days=days,
            times=times,
            area=area,
            data_format=data_format,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
