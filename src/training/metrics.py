"""Training and evaluation metrics for four-channel weather targets."""

from __future__ import annotations

import torch


TARGET_NAMES = ["temperature", "precipitation", "u_wind", "v_wind"]


def _expanded_node_mask(node_mask: torch.Tensor | None, prediction: torch.Tensor) -> torch.Tensor | None:
    if node_mask is None:
        return None
    node_mask = node_mask.to(device=prediction.device, dtype=torch.bool)
    if node_mask.ndim == prediction.ndim - 2:
        while node_mask.ndim < prediction.ndim - 1:
            node_mask = node_mask.unsqueeze(0)
        return node_mask.unsqueeze(-1).expand_as(prediction)
    if node_mask.ndim == prediction.ndim - 1:
        return node_mask.unsqueeze(-1).expand_as(prediction)
    if node_mask.ndim == prediction.ndim and node_mask.shape == prediction.shape:
        return node_mask
    raise ValueError(f"node_mask shape {tuple(node_mask.shape)} is incompatible with prediction shape {tuple(prediction.shape)}")


def _masked_channel_metric(prediction: torch.Tensor, target: torch.Tensor, fn, node_mask: torch.Tensor | None = None) -> torch.Tensor:
    mask = torch.isfinite(prediction) & torch.isfinite(target)
    region_mask = _expanded_node_mask(node_mask, prediction)
    if region_mask is not None:
        mask = mask & region_mask
    values = []
    for channel in range(prediction.shape[-1]):
        channel_mask = mask[..., channel]
        if channel_mask.any():
            values.append(fn(prediction[..., channel][channel_mask], target[..., channel][channel_mask]))
        else:
            values.append(prediction.new_tensor(float("nan")))
    return torch.stack(values)


def rmse_per_channel(prediction: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
    return _masked_channel_metric(prediction, target, lambda p, t: torch.sqrt(((p - t) ** 2).mean()), node_mask=node_mask)


def mae_per_channel(prediction: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
    return _masked_channel_metric(prediction, target, lambda p, t: (p - t).abs().mean(), node_mask=node_mask)


def bias_per_channel(prediction: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
    return _masked_channel_metric(prediction, target, lambda p, t: (p - t).mean(), node_mask=node_mask)


def summarize_metrics(prediction: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor | None = None) -> dict[str, float]:
    rmse = rmse_per_channel(prediction, target, node_mask=node_mask)
    mae = mae_per_channel(prediction, target, node_mask=node_mask)
    bias = bias_per_channel(prediction, target, node_mask=node_mask)
    out: dict[str, float] = {}
    for idx, name in enumerate(TARGET_NAMES):
        out[f"{name}_rmse"] = float(rmse[idx])
        out[f"{name}_mae"] = float(mae[idx])
        out[f"{name}_bias"] = float(bias[idx])
    return out
