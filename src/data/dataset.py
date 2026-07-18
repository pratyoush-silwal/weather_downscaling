"""Dataset utilities for static-graph, dynamic-weather training samples."""

from __future__ import annotations

from bisect import bisect_right
import calendar
from pathlib import Path
from typing import Sequence
from zipfile import is_zipfile

import torch
from torch.utils.data import Dataset


FULL_FEATURE_NAMES = [
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


def _load_torch(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _static_x(graph: dict) -> torch.Tensor:
    if "static_x" in graph:
        return graph["static_x"].float()
    if "x_raw" in graph:
        feature_names = graph.get("metadata", {}).get("node_feature_names", [])
        if "elevation_m" in feature_names:
            start = feature_names.index("elevation_m")
            return graph["x_raw"][:, start : start + 3].float()
        return graph["x_raw"][:, -5:-2].float()
    if all(key in graph for key in ["elevation", "slope", "aspect"]):
        return torch.stack(
            [graph["elevation"], graph["slope"], graph["aspect"]],
            dim=1,
        ).float()
    raise ValueError("Static graph needs static_x or elevation/slope/aspect tensors")


def _month_length_from_path(path: Path) -> int | None:
    stem = path.stem
    yyyymm = stem.rsplit("_", 1)[-1]
    if len(yyyymm) != 6 or not yyyymm.isdigit():
        return None
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    if not 1 <= month <= 12:
        return None
    return calendar.monthrange(year, month)[1] * 24


class WeatherGraphDataset(Dataset):
    """Return one weather graph sample per timestep.

    Static topology and terrain are loaded once from ``static_graph_path``.
    Dynamic monthly ERA5 packages are loaded lazily and cached one file at a
    time. Each item returns a plain dictionary so the code works without
    torch_geometric:

        x:          [N, 11]
        edge_index: [2, E]
        edge_attr:  [E, C_edge]
        y:          optional [N, C_target]
        pos:        [N, 2]
        timestamp:  str
    """

    def __init__(
        self,
        static_graph_path: str | Path,
        dynamic_paths: str | Path | Sequence[str | Path],
        target_paths: str | Path | Sequence[str | Path] | None = None,
        skip_invalid_dynamic: bool = True,
        infer_full_month_lengths: bool = True,
    ) -> None:
        self.static_graph_path = Path(static_graph_path)
        self.graph = _load_torch(self.static_graph_path)
        self.static_x = _static_x(self.graph)
        self.edge_index = self.graph["edge_index"].long()
        self.edge_attr = self.graph["edge_attr"].float()
        self.pos = self.graph["pos"].float()

        if isinstance(dynamic_paths, (str, Path)):
            path = Path(dynamic_paths)
            if path.is_dir():
                self.dynamic_paths = sorted(path.glob("era5_dynamic_*.pt"))
            else:
                self.dynamic_paths = [path]
        else:
            self.dynamic_paths = [Path(path) for path in dynamic_paths]
        if skip_invalid_dynamic:
            invalid = [path for path in self.dynamic_paths if not is_zipfile(path)]
            if invalid:
                print(f"Skipping {len(invalid)} invalid dynamic tensor file(s)")
                self.dynamic_paths = [path for path in self.dynamic_paths if path not in invalid]
        if not self.dynamic_paths:
            raise ValueError("No dynamic ERA5 tensor files were provided")

        if target_paths is None:
            self.target_paths: list[Path] | None = None
        elif isinstance(target_paths, (str, Path)):
            target_path = Path(target_paths)
            self.target_paths = sorted(target_path.glob("*.pt")) if target_path.is_dir() else [target_path]
        else:
            self.target_paths = [Path(path) for path in target_paths]
        if self.target_paths is not None and len(self.target_paths) != len(self.dynamic_paths):
            raise ValueError("target_paths must match dynamic_paths one-to-one")

        self._lengths: list[int] = []
        self._cumulative: list[int] = []
        total = 0
        for path in self.dynamic_paths:
            length = _month_length_from_path(path) if infer_full_month_lengths else None
            if length is None:
                package = _load_torch(path)
                if "x_dynamic" not in package or "time_features" not in package:
                    raise ValueError(f"{path} is not an era5_monthly_dynamic_tensor package")
                length = int(package["x_dynamic"].shape[0])
                if package["x_dynamic"].shape[1] != self.static_x.shape[0]:
                    raise ValueError(f"{path} node count does not match static graph")
            self._lengths.append(length)
            total += length
            self._cumulative.append(total)

        self._cache_index: int | None = None
        self._cache_dynamic: dict | None = None
        self._cache_target: dict | None = None

    def __len__(self) -> int:
        return self._cumulative[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = bisect_right(self._cumulative, index)
        previous = self._cumulative[file_index - 1] if file_index else 0
        return file_index, index - previous

    def _load_month(self, file_index: int) -> tuple[dict, dict | None]:
        if self._cache_index != file_index:
            self._cache_dynamic = _load_torch(self.dynamic_paths[file_index])
            if self._cache_dynamic["x_dynamic"].shape[1] != self.static_x.shape[0]:
                raise ValueError(f"{self.dynamic_paths[file_index]} node count does not match static graph")
            self._cache_target = (
                _load_torch(self.target_paths[file_index]) if self.target_paths is not None else None
            )
            self._cache_index = file_index
        return self._cache_dynamic, self._cache_target

    def __getitem__(self, index: int) -> dict:
        file_index, local_index = self._locate(index)
        dynamic, target = self._load_month(file_index)

        x_dynamic = dynamic["x_dynamic"][local_index].float()
        time_features = dynamic["time_features"][local_index].float()
        temporal_x = time_features.unsqueeze(0).expand(self.static_x.shape[0], -1)
        x = torch.cat([x_dynamic, self.static_x, temporal_x], dim=1)

        sample = {
            "x": x,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "pos": self.pos,
            "time_index": torch.tensor(index, dtype=torch.long),
            "timestamp": dynamic["timestamps"][local_index],
            "feature_names": FULL_FEATURE_NAMES,
            "dynamic_path": str(self.dynamic_paths[file_index]),
        }
        if target is not None:
            sample["y"] = target["y"][local_index].float()
        return sample
