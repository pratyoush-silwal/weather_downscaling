"""Validated physical-reference normalization (no fitted data statistics)."""

import torch


def affine_parameters(channels, offsets=None, scales=None):
    offset = torch.tensor(offsets if offsets is not None else [0.] * channels, dtype=torch.float32)
    scale = torch.tensor(scales if scales is not None else [1.] * channels, dtype=torch.float32)
    if offset.shape != (channels,) or scale.shape != (channels,):
        raise ValueError("Normalization parameters must match the channel count")
    if not torch.isfinite(offset).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Normalization requires finite offsets and positive finite scales")
    return offset, scale
