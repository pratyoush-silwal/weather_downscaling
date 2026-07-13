"""Loss functions for supervised and physics-informed weather downscaling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def finite_mask(*tensors: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(tensors[0], dtype=torch.bool)
    for tensor in tensors:
        mask = mask & torch.isfinite(tensor)
    return mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = finite_mask(prediction, target)
    error = (prediction - torch.nan_to_num(target, nan=0.0)) ** 2
    if channel_weights is not None:
        weights = channel_weights.to(device=prediction.device, dtype=prediction.dtype)
        error = error * weights.view(*([1] * (error.ndim - 1)), -1)
    if not mask.any():
        return prediction.new_tensor(0.0)
    return error[mask].mean()


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = finite_mask(prediction, target)
    error = (prediction - torch.nan_to_num(target, nan=0.0)).abs()
    if channel_weights is not None:
        weights = channel_weights.to(device=prediction.device, dtype=prediction.dtype)
        error = error * weights.view(*([1] * (error.ndim - 1)), -1)
    if not mask.any():
        return prediction.new_tensor(0.0)
    return error[mask].mean()


def _ensure_batched_node_values(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if values.ndim == 1:
        return values.unsqueeze(0), False
    if values.ndim == 2:
        return values, True
    raise ValueError(f"Expected [N] or [B,N], got {tuple(values.shape)}")


def lapse_rate_loss(
    prediction: torch.Tensor,
    edge_index: torch.Tensor,
    elevation: torch.Tensor,
    temperature_channel: int = 0,
    lapse_rate_k_per_m: float = -0.0065,
    min_elevation_delta_m: float = 25.0,
    loss: str = "huber",
) -> torch.Tensor:
    """Penalize temperature differences inconsistent with elevation changes.

    For an edge ``src -> dst`` the expected local temperature difference is:

        T_dst - T_src ~= lapse_rate * (z_dst - z_src)

    This is a soft regularizer, not a hard physical law. It should usually have
    a small weight because humidity, radiation, valley inversions, and synoptic
    flow can violate a simple lapse-rate assumption.
    """

    temp = prediction[..., temperature_channel]
    temp_batched, _ = _ensure_batched_node_values(temp)
    elev = elevation.to(device=prediction.device, dtype=prediction.dtype)
    src = edge_index[0].to(prediction.device).long()
    dst = edge_index[1].to(prediction.device).long()

    dz = elev[dst] - elev[src]
    valid = dz.abs() >= min_elevation_delta_m
    if not valid.any():
        return prediction.new_tensor(0.0)

    actual = temp_batched[:, dst] - temp_batched[:, src]
    expected = lapse_rate_k_per_m * dz.view(1, -1)
    residual = actual[:, valid] - expected[:, valid]
    if loss == "mse":
        return residual.pow(2).mean()
    if loss == "mae":
        return residual.abs().mean()
    if loss == "huber":
        return F.smooth_l1_loss(residual, torch.zeros_like(residual))
    raise ValueError(f"Unsupported lapse-rate loss: {loss}")


def _ensure_batched_nodes(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if values.ndim == 2:
        return values.unsqueeze(0), False
    if values.ndim == 3:
        return values, True
    raise ValueError(f"Expected [N,C] or [B,N,C], got {tuple(values.shape)}")


def _edge_displacements_m(pos: torch.Tensor, edge_index: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    src = edge_index[0].to(pos.device).long()
    dst = edge_index[1].to(pos.device).long()
    lat = pos[:, 0].to(dtype=dtype)
    lon = pos[:, 1].to(dtype=dtype)
    mean_lat_rad = torch.deg2rad((lat[src] + lat[dst]) * 0.5)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * torch.cos(mean_lat_rad)
    dx = (lon[dst] - lon[src]) * meters_per_deg_lon
    dy = (lat[dst] - lat[src]) * meters_per_deg_lat
    return dx, dy


def graph_divergence_loss(
    prediction: torch.Tensor,
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    u_channel: int = 2,
    v_channel: int = 3,
    eps: float = 1.0,
) -> torch.Tensor:
    """Softly penalize horizontal wind divergence on the graph.

    This approximates divergence from edge-wise directional derivatives:

        ((u_j - u_i) * dx + (v_j - v_i) * dy) / (dx^2 + dy^2)

    then averages outgoing edge contributions per source node. It is a weak
    regularizer, not a strict incompressibility constraint.
    """

    if max(u_channel, v_channel) >= prediction.shape[-1]:
        return prediction.new_tensor(0.0)

    pred, _ = _ensure_batched_nodes(prediction)
    edge_index = edge_index.to(prediction.device)
    pos = pos.to(device=prediction.device, dtype=prediction.dtype)
    src = edge_index[0].long()
    dst = edge_index[1].long()

    dx, dy = _edge_displacements_m(pos, edge_index, prediction.dtype)
    dist_sq = (dx * dx + dy * dy).clamp_min(eps)

    du = pred[:, dst, u_channel] - pred[:, src, u_channel]
    dv = pred[:, dst, v_channel] - pred[:, src, v_channel]
    edge_divergence = (du * dx.view(1, -1) + dv * dy.view(1, -1)) / dist_sq.view(1, -1)

    batch_size, node_count = pred.shape[0], pred.shape[1]
    node_divergence = pred.new_zeros(batch_size, node_count)
    node_divergence.index_add_(1, src, edge_divergence)
    degree = torch.bincount(src, minlength=node_count).to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    node_divergence = node_divergence / degree.clamp_min(1.0).view(1, -1)
    return node_divergence.pow(2).mean()


@dataclass(frozen=True)
class LossWeights:
    data: float = 1.0
    lapse_rate: float = 0.0
    divergence: float = 0.0


class PIGNNLoss(nn.Module):
    """Combined supervised and physics-informed loss.

    ``target`` may contain NaNs; supervised loss ignores missing target values.
    The physics terms only require predictions, graph edges, and elevation.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        data_loss: str = "mse",
        channel_weights: tuple[float, ...] | None = None,
        temperature_channel: int = 0,
        u_channel: int = 2,
        v_channel: int = 3,
        lapse_rate_k_per_m: float = -0.0065,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.data_loss = data_loss
        self.temperature_channel = temperature_channel
        self.u_channel = u_channel
        self.v_channel = v_channel
        self.lapse_rate_k_per_m = lapse_rate_k_per_m
        if channel_weights is None:
            self.register_buffer("channel_weights", None)
        else:
            self.register_buffer("channel_weights", torch.tensor(channel_weights, dtype=torch.float32))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None,
        edge_index: torch.Tensor,
        elevation: torch.Tensor,
        pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        components: dict[str, torch.Tensor] = {}

        if target is None or self.weights.data == 0:
            data = prediction.new_tensor(0.0)
        elif self.data_loss == "mse":
            data = masked_mse(prediction, target, self.channel_weights)
        elif self.data_loss == "mae":
            data = masked_mae(prediction, target, self.channel_weights)
        else:
            raise ValueError(f"Unsupported data loss: {self.data_loss}")
        components["data"] = data

        components["lapse_rate"] = lapse_rate_loss(
            prediction=prediction,
            edge_index=edge_index,
            elevation=elevation,
            temperature_channel=self.temperature_channel,
            lapse_rate_k_per_m=self.lapse_rate_k_per_m,
        )
        components["divergence"] = graph_divergence_loss(
            prediction=prediction,
            edge_index=edge_index,
            pos=pos,
            u_channel=self.u_channel,
            v_channel=self.v_channel,
        )

        total = (
            self.weights.data * components["data"]
            + self.weights.lapse_rate * components["lapse_rate"]
            + self.weights.divergence * components["divergence"]
        )
        components["total"] = total
        return components
