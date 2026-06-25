#!/usr/bin/env python3
"""Create an interactive 3D prism visualization from the Nepal SRTM GeoTIFF.

The output is a standalone HTML file that renders a sampled DEM as square-base
prisms. It uses Three.js from a CDN when the HTML is opened in a browser.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


TIFF_TYPES = {
    1: ("B", 1),
    2: ("c", 1),
    3: ("H", 2),
    4: ("I", 4),
    5: ("II", 8),
    11: ("f", 4),
    12: ("d", 8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive 3D square-prism DEM viewer."
    )
    parser.add_argument(
        "--input",
        default="data/raw/srtm_nepal.tif",
        help="Input GeoTIFF path. Defaults to data/raw/srtm_nepal.tif.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/srtm_nepal_3d.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=180,
        help="Approximate maximum grid dimension to render. Higher is heavier.",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=0.018,
        help="Height multiplier for prism elevation.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=1.0,
        help="Minimum visible prism height.",
    )
    parser.add_argument(
        "--max-prisms",
        type=int,
        default=45000,
        help="Safety cap for rendered prisms.",
    )
    return parser.parse_args()


def read_ifd_entry(raw: bytes, endian: str) -> tuple[int, int, int, bytes]:
    tag, type_id, count, value = struct.unpack(endian + "HHI4s", raw)
    return tag, type_id, count, value


def decode_inline_value(type_id: int, count: int, value: bytes, endian: str) -> int | str:
    fmt, size = TIFF_TYPES[type_id]
    raw = value[: count * size]
    if type_id == 2:
        return raw.rstrip(b"\x00").decode("ascii")
    values = struct.unpack(endian + fmt * count, raw)
    return values[0] if count == 1 else values


def read_tiff_tags(path: Path) -> tuple[str, dict[int, tuple[int, int, int | str]]]:
    with path.open("rb") as handle:
        byte_order = handle.read(2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError("Not a classic TIFF file")

        version = struct.unpack(endian + "H", handle.read(2))[0]
        if version != 42:
            raise ValueError("Only classic TIFF files are supported")

        ifd_offset = struct.unpack(endian + "I", handle.read(4))[0]
        handle.seek(ifd_offset)
        entry_count = struct.unpack(endian + "H", handle.read(2))[0]
        tags: dict[int, tuple[int, int, int | str]] = {}

        for _ in range(entry_count):
            tag, type_id, count, raw_value = read_ifd_entry(handle.read(12), endian)
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


def value(tags: dict[int, tuple[int, int, int | str]], tag: int) -> int | str | tuple:
    if tag not in tags:
        raise ValueError(f"Required TIFF tag {tag} is missing")
    return tags[tag][2]


def sample_geotiff(path: Path, grid: int, max_prisms: int) -> dict[str, object]:
    endian, tags = read_tiff_tags(path)
    width = int(value(tags, 256))
    height = int(value(tags, 257))
    bits = int(value(tags, 258))
    sample_format = int(value(tags, 339))
    rows_per_strip = int(value(tags, 278))
    strip_offsets = value(tags, 273)
    strip_byte_counts = value(tags, 279)
    nodata_raw = tags.get(42113, (None, None, "-32768"))[2]
    nodata = int(str(nodata_raw))

    if bits != 16 or sample_format != 2:
        raise ValueError("This viewer currently expects signed 16-bit integer DEM data")

    if isinstance(strip_offsets, int):
        strip_offsets = (strip_offsets,)
    if isinstance(strip_byte_counts, int):
        strip_byte_counts = (strip_byte_counts,)

    step = max(1, math.ceil(max(width, height) / grid))
    sampled_rows = list(range(0, height, step))
    sampled_cols = list(range(0, width, step))

    while len(sampled_rows) * len(sampled_cols) > max_prisms:
        step += 1
        sampled_rows = list(range(0, height, step))
        sampled_cols = list(range(0, width, step))

    col_idx = np.array(sampled_cols, dtype=np.int64)
    sampled = np.full((len(sampled_rows), len(sampled_cols)), nodata, dtype=np.int16)
    row_to_output = {row: idx for idx, row in enumerate(sampled_rows)}
    sampled_row_set = set(sampled_rows)

    with path.open("rb") as handle:
        for strip_index, offset in enumerate(strip_offsets):
            first_row = strip_index * rows_per_strip
            row_count = min(rows_per_strip, height - first_row)
            wanted_rows = [
                row for row in sampled_rows if first_row <= row < first_row + row_count
            ]
            if not wanted_rows:
                continue

            handle.seek(int(offset))
            raw = handle.read(int(strip_byte_counts[strip_index]))
            strip = np.frombuffer(raw, dtype=endian + "i2").reshape(row_count, width)

            for row in wanted_rows:
                local_row = row - first_row
                sampled[row_to_output[row], :] = strip[local_row, col_idx]

    mask = sampled != nodata
    valid = sampled[mask]
    if valid.size == 0:
        raise ValueError("No valid elevation values were found")

    pixel_scale = tags.get(33550, (None, None, None))[2]
    tiepoint = tags.get(33922, (None, None, None))[2]
    if isinstance(pixel_scale, tuple) and isinstance(tiepoint, tuple):
        lon_min = float(tiepoint[3])
        lat_max = float(tiepoint[4])
        pixel_width = float(pixel_scale[0])
        pixel_height = float(pixel_scale[1])
        lon_max = lon_min + (width - 1) * pixel_width
        lat_min = lat_max - (height - 1) * pixel_height
    else:
        lon_min = lon_max = lat_min = lat_max = None

    return {
        "width": width,
        "height": height,
        "step": step,
        "rows": len(sampled_rows),
        "cols": len(sampled_cols),
        "nodata": nodata,
        "minElevation": int(valid.min()),
        "maxElevation": int(valid.max()),
        "meanElevation": float(valid.mean()),
        "bounds": {
            "lonMin": lon_min,
            "lonMax": lon_max,
            "latMin": lat_min,
            "latMax": lat_max,
        },
        "elevations": sampled.astype(int).tolist(),
    }


def html_document(payload: dict[str, object], vertical_scale: float, base_height: float) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SRTM Nepal 3D Prism Viewer</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #101216;
      color: #f2f4f8;
      font-family: Arial, sans-serif;
    }}
    #scene {{
      position: fixed;
      inset: 0;
    }}
    #hud {{
      position: fixed;
      left: 16px;
      top: 16px;
      max-width: 360px;
      padding: 12px 14px;
      border: 1px solid rgba(255,255,255,.16);
      background: rgba(13,16,22,.82);
      backdrop-filter: blur(8px);
      font-size: 13px;
      line-height: 1.45;
    }}
    #hud h1 {{
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 700;
    }}
    #hud dl {{
      display: grid;
      grid-template-columns: 112px 1fr;
      gap: 4px 10px;
      margin: 0;
    }}
    #hud dt {{
      color: #aeb7c8;
    }}
    #hud dd {{
      margin: 0;
    }}
    #error {{
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      padding: 24px;
      color: white;
      background: #111;
      font: 15px/1.5 Arial, sans-serif;
      text-align: center;
    }}
  </style>
</head>
<body>
  <canvas id="scene"></canvas>
  <aside id="hud">
    <h1>SRTM Nepal 3D Prism Viewer</h1>
    <dl>
      <dt>Rendered grid</dt><dd id="grid"></dd>
      <dt>Sample step</dt><dd id="step"></dd>
      <dt>Elevation range</dt><dd id="range"></dd>
      <dt>Controls</dt><dd>drag rotate, wheel zoom, right-drag pan</dd>
    </dl>
  </aside>
  <div id="error"></div>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.165.0/build/three.module.js"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from 'https://unpkg.com/three@0.165.0/build/three.module.js';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js';

    const dem = {data_json};
    const verticalScale = {vertical_scale};
    const baseHeight = {base_height};
    const canvas = document.getElementById('scene');
    const error = document.getElementById('error');

    document.getElementById('grid').textContent = `${{dem.cols}} x ${{dem.rows}} prisms`;
    document.getElementById('step').textContent = `${{dem.step}} source pixels`;
    document.getElementById('range').textContent = `${{dem.minElevation}} m to ${{dem.maxElevation}} m`;

    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x101216, 1);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x101216, 320, 820);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.set(0, 190, 360);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 20, 0);
    controls.maxDistance = 900;

    const ambient = new THREE.HemisphereLight(0xdce8ff, 0x1d231b, 2.0);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xffffff, 2.8);
    sun.position.set(-140, 240, 90);
    scene.add(sun);

    const rows = dem.rows;
    const cols = dem.cols;
    const spacing = 2.25;
    const cellSize = 1.9;
    const validCells = [];

    for (let r = 0; r < rows; r++) {{
      for (let c = 0; c < cols; c++) {{
        const elevation = dem.elevations[r][c];
        if (elevation !== dem.nodata) {{
          validCells.push([r, c, elevation]);
        }}
      }}
    }}

    const geometry = new THREE.BoxGeometry(cellSize, 1, cellSize);
    const material = new THREE.MeshStandardMaterial({{
      roughness: 0.72,
      metalness: 0.0,
      vertexColors: true
    }});
    const mesh = new THREE.InstancedMesh(geometry, material, validCells.length);
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    const low = dem.minElevation;
    const high = dem.maxElevation;
    const span = Math.max(1, high - low);
    const xOffset = (cols - 1) * spacing / 2;
    const zOffset = (rows - 1) * spacing / 2;

    for (let i = 0; i < validCells.length; i++) {{
      const [r, c, elevation] = validCells[i];
      const height = Math.max(baseHeight, Math.max(0, elevation - low) * verticalScale + baseHeight);
      dummy.position.set(c * spacing - xOffset, height / 2, r * spacing - zOffset);
      dummy.scale.set(1, height, 1);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      const t = Math.max(0, Math.min(1, (elevation - low) / span));
      if (t < 0.28) {{
        color.setRGB(0.05 + t * 0.45, 0.28 + t * 0.9, 0.18);
      }} else if (t < 0.58) {{
        color.setRGB(0.35 + t * 0.35, 0.40 + t * 0.35, 0.22);
      }} else if (t < 0.82) {{
        color.setRGB(0.48 + t * 0.32, 0.42 + t * 0.28, 0.34 + t * 0.16);
      }} else {{
        color.setRGB(0.78 + t * 0.22, 0.82 + t * 0.18, 0.88 + t * 0.12);
      }}
      mesh.setColorAt(i, color);
    }}
    scene.add(mesh);

    const baseGeometry = new THREE.PlaneGeometry(cols * spacing, rows * spacing);
    const baseMaterial = new THREE.MeshStandardMaterial({{
      color: 0x1b2220,
      roughness: 0.9,
      metalness: 0
    }});
    const base = new THREE.Mesh(baseGeometry, baseMaterial);
    base.rotation.x = -Math.PI / 2;
    base.position.y = -0.03;
    scene.add(base);

    function resize() {{
      const width = window.innerWidth;
      const height = window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }}

    function animate() {{
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}

    window.addEventListener('resize', resize);
    animate();
  </script>
  <script nomodule>
    const error = document.getElementById('error');
    error.style.display = 'grid';
    error.textContent = 'This viewer needs a modern browser with JavaScript modules enabled.';
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sample_geotiff(input_path, args.grid, args.max_prisms)
    output_path.write_text(
        html_document(payload, args.vertical_scale, args.base_height),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    print(
        f"Rendered grid: {payload['cols']} x {payload['rows']} "
        f"({payload['cols'] * payload['rows']} sampled cells)"
    )
    print(
        f"Elevation range: {payload['minElevation']} m to {payload['maxElevation']} m"
    )


if __name__ == "__main__":
    main()
