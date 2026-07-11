"""Model architecture package for PI-GNN weather downscaling."""

from .layers import EdgeConditionedMessagePassing, MLP, PositivePrecipitationHead, ResidualGraphBlock
from .losses import LossWeights, PIGNNLoss, edge_smoothness_loss, lapse_rate_loss, masked_mae, masked_mse
from .piggn import PIGNN, PIGNNConfig, build_pignn_from_config

__all__ = [
    "EdgeConditionedMessagePassing",
    "LossWeights",
    "MLP",
    "PIGNN",
    "PIGNNConfig",
    "PIGNNLoss",
    "PositivePrecipitationHead",
    "ResidualGraphBlock",
    "build_pignn_from_config",
    "edge_smoothness_loss",
    "lapse_rate_loss",
    "masked_mae",
    "masked_mse",
]
