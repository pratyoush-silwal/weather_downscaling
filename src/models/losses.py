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


def precipitation_nonnegative_loss(
    prediction: torch.Tensor,
    precipitation_channel: int = 1,
) -> torch.Tensor:
    if precipitation_channel >= prediction.shape[-1]:
        return prediction.new_tensor(0.0)
    precip = prediction[..., precipitation_channel]
    return F.relu(-precip).pow(2).mean()


def edge_smoothness_loss(
    prediction: torch.Tensor,
    edge_index: torch.Tensor,
    channels: tuple[int, ...] = (0, 1),
    huber: bool = True,
) -> torch.Tensor:
    values = prediction[..., list(channels)]
    if values.ndim == 2:
        values = values.unsqueeze(0)
    src = edge_index[0].to(prediction.device).long()
    dst = edge_index[1].to(prediction.device).long()
    diff = values[:, dst, :] - values[:, src, :]
    if huber:
        return F.smooth_l1_loss(diff, torch.zeros_like(diff))
    return diff.pow(2).mean()


@dataclass(frozen=True)
class LossWeights:
    data: float = 1.0
    lapse_rate: float = 0.0
    precipitation_nonnegative: float = 0.0
    smoothness: float = 0.0


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
        precipitation_channel: int = 1,
        lapse_rate_k_per_m: float = -0.0065,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.data_loss = data_loss
        self.temperature_channel = temperature_channel
        self.precipitation_channel = precipitation_channel
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
        components["precipitation_nonnegative"] = precipitation_nonnegative_loss(
            prediction=prediction,
            precipitation_channel=self.precipitation_channel,
        )
        components["smoothness"] = edge_smoothness_loss(prediction, edge_index)

        total = (
            self.weights.data * components["data"]
            + self.weights.lapse_rate * components["lapse_rate"]
            + self.weights.precipitation_nonnegative * components["precipitation_nonnegative"]
            + self.weights.smoothness * components["smoothness"]
        )
        components["total"] = total
        return components
