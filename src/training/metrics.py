"""Training and evaluation metrics for four-channel weather targets."""

from __future__ import annotations

import torch


TARGET_NAMES = ["temperature", "precipitation", "u_wind", "v_wind"]


def _masked_channel_metric(prediction: torch.Tensor, target: torch.Tensor, fn) -> torch.Tensor:
    mask = torch.isfinite(prediction) & torch.isfinite(target)
    values = []
    for channel in range(prediction.shape[-1]):
        channel_mask = mask[..., channel]
        if channel_mask.any():
            values.append(fn(prediction[..., channel][channel_mask], target[..., channel][channel_mask]))
        else:
            values.append(prediction.new_tensor(float("nan")))
    return torch.stack(values)


def rmse_per_channel(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _masked_channel_metric(prediction, target, lambda p, t: torch.sqrt(((p - t) ** 2).mean()))


def mae_per_channel(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _masked_channel_metric(prediction, target, lambda p, t: (p - t).abs().mean())


def bias_per_channel(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _masked_channel_metric(prediction, target, lambda p, t: (p - t).mean())


def summarize_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    rmse = rmse_per_channel(prediction, target)
    mae = mae_per_channel(prediction, target)
    bias = bias_per_channel(prediction, target)
    out: dict[str, float] = {}
    for idx, name in enumerate(TARGET_NAMES):
        out[f"{name}_rmse"] = float(rmse[idx])
        out[f"{name}_mae"] = float(mae[idx])
        out[f"{name}_bias"] = float(bias[idx])
    return out
