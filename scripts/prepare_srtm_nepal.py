#!/usr/bin/env python3
"""Download and mosaic SRTM tiles for the configured Nepal project extent."""

from __future__ import annotations

import gzip
import json
import math
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


LAT_MIN = 26
LAT_MAX = 31
LON_MIN = 80
LON_MAX = 89
SAMPLES_PER_TILE = 3601
SAMPLES_PER_DEGREE = 3600
NODATA = -32768
BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TILE_DIR = RAW_DIR / "srtm_tiles"
MOSAIC_DAT = RAW_DIR / "srtm_nepal_int16.dat"
MOSAIC_TIF = RAW_DIR / "srtm_nepal.tif"
MANIFEST = RAW_DIR / "srtm_nepal_manifest.json"


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def tile_url(name: str) -> str:
    return f"{BASE_URL}/{name[:3]}/{name}.hgt.gz"


def download(url: str, dest: Path, attempts: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return

    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            partial.replace(dest)
            return
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def download_tiles(tile_records: list[dict[str, str]]) -> None:
    def fetch(record: dict[str, str]) -> str:
        download(record["url"], ROOT / record["path"])
        return record["name"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, record) for record in tile_records]
        for future in as_completed(futures):
            print(f"Ready {future.result()}", flush=True)


def read_hgt_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    expected = SAMPLES_PER_TILE * SAMPLES_PER_TILE * 2
    if len(data) != expected:
        raise ValueError(f"{path.name} has {len(data)} bytes, expected {expected}")
    return np.frombuffer(data, dtype=">i2").reshape(SAMPLES_PER_TILE, SAMPLES_PER_TILE)


def write_manifest(tiles: list[dict[str, str]]) -> None:
    payload = {
        "source": "Mapzen/AWS public elevation-tiles-prod Skadi SRTM HGT tiles",
        "source_base_url": BASE_URL,
        "created_for_bounds": {
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "crs": "EPSG:4326",
        },
        "resolution_degrees": 1 / SAMPLES_PER_DEGREE,
        "output": str(MOSAIC_TIF.relative_to(ROOT)),
        "tiles": tiles,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_mosaic(tile_paths: dict[tuple[int, int], Path]) -> tuple[int, int]:
    rows = (LAT_MAX - LAT_MIN) * SAMPLES_PER_DEGREE + 1
    cols = (LON_MAX - LON_MIN) * SAMPLES_PER_DEGREE + 1
    mosaic = np.memmap(MOSAIC_DAT, dtype=np.int16, mode="w+", shape=(rows, cols))
    mosaic[:] = NODATA

    for lat in range(LAT_MAX - 1, LAT_MIN - 1, -1):
        for lon in range(LON_MIN, LON_MAX):
            name = tile_name(lat, lon)
            print(f"Mosaicking {name}", flush=True)
            tile = read_hgt_gz(tile_paths[(lat, lon)]).astype(np.int16, copy=False)
            row = (LAT_MAX - (lat + 1)) * SAMPLES_PER_DEGREE
            col = (lon - LON_MIN) * SAMPLES_PER_DEGREE
            mosaic[row : row + SAMPLES_PER_TILE, col : col + SAMPLES_PER_TILE] = tile

    mosaic.flush()
    return rows, cols


def pack_ifd_entry(tag: int, type_id: int, count: int, value: int | bytes, endian: str) -> bytes:
    if isinstance(value, bytes):
        raw = value.ljust(4, b"\x00")
        return struct.pack(endian + "HHI4s", tag, type_id, count, raw)
    return struct.pack(endian + "HHII", tag, type_id, count, value)


def write_geotiff(rows: int, cols: int) -> None:
    endian = "<"
    rows_per_strip = 64
    strip_rows = [min(rows_per_strip, rows - start) for start in range(0, rows, rows_per_strip)]
    strip_byte_counts = [count * cols * 2 for count in strip_rows]
    strip_count = len(strip_rows)

    geo_key_directory = np.array(
        [
            1,
            1,
            0,
            4,
            1024,
            0,
            1,
            2,
            1025,
            0,
            1,
            2,
            2048,
            0,
            1,
            4326,
            2054,
            0,
            1,
            9102,
        ],
        dtype="<u2",
    ).tobytes()
    pixel_scale = struct.pack(endian + "ddd", 1 / SAMPLES_PER_DEGREE, 1 / SAMPLES_PER_DEGREE, 0.0)
    tiepoint = struct.pack(endian + "dddddd", 0.0, 0.0, 0.0, float(LON_MIN), float(LAT_MAX), 0.0)
    nodata = f"{NODATA}\x00".encode("ascii")
    strip_offsets_placeholder = b"\x00" * (strip_count * 4)
    strip_byte_counts_raw = struct.pack(endian + f"{strip_count}I", *strip_byte_counts)

    value_blobs: list[tuple[str, bytes]] = [
        ("strip_offsets", strip_offsets_placeholder),
        ("strip_byte_counts", strip_byte_counts_raw),
        ("pixel_scale", pixel_scale),
        ("tiepoint", tiepoint),
        ("geo_key_directory", geo_key_directory),
        ("nodata", nodata),
    ]

    tags = [
        (256, 4, 1, cols),
        (257, 4, 1, rows),
        (258, 3, 1, struct.pack(endian + "H", 16)),
        (259, 3, 1, struct.pack(endian + "H", 1)),
        (262, 3, 1, struct.pack(endian + "H", 1)),
        (273, 4, strip_count, "strip_offsets"),
        (277, 3, 1, struct.pack(endian + "H", 1)),
        (278, 4, 1, rows_per_strip),
        (279, 4, strip_count, "strip_byte_counts"),
        (284, 3, 1, struct.pack(endian + "H", 1)),
        (339, 3, 1, struct.pack(endian + "H", 2)),
        (33550, 12, 3, "pixel_scale"),
        (33922, 12, 6, "tiepoint"),
        (34735, 3, len(geo_key_directory) // 2, "geo_key_directory"),
        (42113, 2, len(nodata), "nodata"),
    ]

    ifd_start = 8
    ifd_size = 2 + len(tags) * 12 + 4
    blob_offsets: dict[str, int] = {}
    cursor = ifd_start + ifd_size
    for name, blob in value_blobs:
        if cursor % 2:
            cursor += 1
        blob_offsets[name] = cursor
        cursor += len(blob)

    data_start = cursor if cursor % 2 == 0 else cursor + 1
    strip_offsets = []
    cursor = data_start
    for byte_count in strip_byte_counts:
        strip_offsets.append(cursor)
        cursor += byte_count
    if cursor >= 2**32:
        raise ValueError("Classic TIFF limit exceeded; BigTIFF writer would be required")

    strip_offsets_raw = struct.pack(endian + f"{strip_count}I", *strip_offsets)
    value_blobs[0] = ("strip_offsets", strip_offsets_raw)

    with MOSAIC_TIF.open("wb") as out:
        out.write(b"II")
        out.write(struct.pack(endian + "H", 42))
        out.write(struct.pack(endian + "I", ifd_start))
        out.write(struct.pack(endian + "H", len(tags)))
        for tag, type_id, count, value in tags:
            if isinstance(value, str):
                out.write(pack_ifd_entry(tag, type_id, count, blob_offsets[value], endian))
            else:
                out.write(pack_ifd_entry(tag, type_id, count, value, endian))
        out.write(struct.pack(endian + "I", 0))

        for name, blob in value_blobs:
            out.seek(blob_offsets[name])
            out.write(blob)

        out.seek(data_start)
        mosaic = np.memmap(MOSAIC_DAT, dtype=np.int16, mode="r", shape=(rows, cols))
        for start_row, count in zip(range(0, rows, rows_per_strip), strip_rows):
            block = np.asarray(mosaic[start_row : start_row + count, :], dtype="<i2")
            out.write(block.tobytes(order="C"))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TILE_DIR.mkdir(parents=True, exist_ok=True)

    tile_records: list[dict[str, str]] = []
    tile_paths: dict[tuple[int, int], Path] = {}
    for lat in range(LAT_MIN, LAT_MAX):
        for lon in range(LON_MIN, LON_MAX):
            name = tile_name(lat, lon)
            url = tile_url(name)
            dest = TILE_DIR / f"{name}.hgt.gz"
            tile_records.append({"name": name, "url": url, "path": str(dest.relative_to(ROOT))})
            tile_paths[(lat, lon)] = dest

    write_manifest(tile_records)
    download_tiles(tile_records)
    rows, cols = build_mosaic(tile_paths)
    write_geotiff(rows, cols)
    MOSAIC_DAT.unlink(missing_ok=True)
    print(f"Wrote {MOSAIC_TIF} ({rows} x {cols})", flush=True)


if __name__ == "__main__":
    main()
