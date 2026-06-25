#!/usr/bin/env python3
"""Visualize the saved Nepal graph network as a lon/lat map."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved torch graph package.")
    parser.add_argument(
        "--graph",
        default="data/processed/nepal_graph.pt",
        help="Input graph package created by src/data/build_graph.py.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/nepal_graph_network.png",
        help="Output image path. Use .png, .jpg, .pdf, or any matplotlib-supported format.",
    )
    parser.add_argument(
        "--color-by",
        default="elevation",
        choices=["elevation", "slope", "aspect", "degree", "node_type"],
        help="Node variable used for coloring.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=0,
        help="Maximum number of edges to draw. Use 0 to draw all edges.",
    )
    parser.add_argument(
        "--node-size",
        type=float,
        default=5.0,
        help="Scatter marker size for graph nodes.",
    )
    parser.add_argument(
        "--edge-alpha",
        type=float,
        default=0.18,
        help="Edge transparency from 0 to 1.",
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=0.28,
        help="Line width for graph edges.",
    )
    parser.add_argument(
        "--edge-color",
        default="#111827",
        help="Matplotlib color for graph edges.",
    )
    parser.add_argument(
        "--figsize",
        default="12,8",
        help="Figure size as width,height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_graph(path: Path) -> dict:
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        graph = torch.load(path, map_location="cpu")
    required = {"pos", "edge_index"}
    missing = required - set(graph)
    if missing:
        raise ValueError(f"{path} is missing required graph keys: {sorted(missing)}")
    return graph


def edge_sample(edge_index: np.ndarray, max_edges: int) -> np.ndarray:
    edge_count = edge_index.shape[1]
    if max_edges <= 0 or edge_count <= max_edges:
        return edge_index
    sample_idx = np.linspace(0, edge_count - 1, max_edges, dtype=np.int64)
    return edge_index[:, sample_idx]


def node_values(graph: dict, color_by: str, edge_index: np.ndarray) -> tuple[np.ndarray, str, str]:
    if color_by in {"elevation", "slope", "aspect"}:
        if color_by not in graph:
            raise ValueError(f"Graph package does not contain '{color_by}'")
        label = {
            "elevation": "Elevation (m)",
            "slope": "Slope (rad)",
            "aspect": "Aspect (rad)",
        }[color_by]
        return as_numpy(graph[color_by]).astype(np.float32), label, "viridis"

    if color_by == "degree":
        node_count = as_numpy(graph["pos"]).shape[0]
        degree = np.bincount(edge_index.reshape(-1), minlength=node_count).astype(np.float32)
        return degree, "Incident degree", "magma"

    metadata = graph.get("metadata", {})
    node_types = metadata.get("node_types")
    if not node_types:
        raise ValueError("Graph metadata does not contain node_types")
    values = np.array([1 if node_type == "station" else 0 for node_type in node_types], dtype=np.float32)
    return values, "Node type: grid=0, station=1", "coolwarm"


def parse_figsize(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--figsize must be formatted as width,height")
    return float(parts[0]), float(parts[1])


def draw_graph(args: argparse.Namespace) -> Path:
    graph_path = resolve_path(args.graph)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = load_graph(graph_path)
    pos = as_numpy(graph["pos"]).astype(np.float32)
    edge_index = as_numpy(graph["edge_index"]).astype(np.int64)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError("Expected pos with shape [nodes, 2] as [lat, lon]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected edge_index with shape [2, edges]")

    lat = pos[:, 0]
    lon = pos[:, 1]
    sampled_edges = edge_sample(edge_index, args.max_edges)
    values, colorbar_label, cmap = node_values(graph, args.color_by, edge_index)

    segments = np.stack(
        [
            np.column_stack([lon[sampled_edges[0]], lat[sampled_edges[0]]]),
            np.column_stack([lon[sampled_edges[1]], lat[sampled_edges[1]]]),
        ],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=parse_figsize(args.figsize), constrained_layout=True)
    ax.add_collection(
        LineCollection(
            segments,
            colors=args.edge_color,
            linewidths=args.edge_width,
            alpha=float(np.clip(args.edge_alpha, 0.0, 1.0)),
            zorder=1,
        )
    )
    scatter = ax.scatter(
        lon,
        lat,
        c=values,
        s=args.node_size,
        cmap=cmap,
        linewidths=0,
        alpha=0.95,
        zorder=2,
    )

    metadata = graph.get("metadata", {})
    region_name = metadata.get("region", {}).get("name", "graph")
    title = (
        f"{region_name.title()} graph network "
        f"({pos.shape[0]:,} nodes, {edge_index.shape[1]:,} edges; "
        f"showing {sampled_edges.shape[1]:,} edges)"
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.5, alpha=0.8)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82, pad=0.02)
    colorbar.set_label(colorbar_label)

    fig.savefig(output_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    output_path = draw_graph(parse_args())
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
