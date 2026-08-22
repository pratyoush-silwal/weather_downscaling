"""Per-node MLP baseline for weather downscaling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .layers import MLP, PositivePrecipitationHead


@dataclass(frozen=True)
class MLPBaselineConfig:
    node_input_channels: int = 11
    hidden_channels: int = 128
    num_layers: int = 3
    output_channels: int = 4
    dropout: float = 0.1
    precipitation_channel: int | None = 1
    coarse_temperature_channel: int = 0
    coarse_precipitation_channel: int = 5
    coarse_u_channel: int = 2
    coarse_v_channel: int = 3
    use_coarse_temperature_residual: bool = True
    use_coarse_precipitation_residual: bool = True
    use_coarse_wind_residual: bool = True


class NodewiseMLPBaseline(nn.Module):
    """No edges, no message passing. Same inputs and outputs as the GNN."""

    def __init__(self, config: MLPBaselineConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = MLPBaselineConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config
        self.backbone = MLP(
            in_channels=config.node_input_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.hidden_channels,
            num_layers=max(1, config.num_layers - 1),
            dropout=config.dropout,
            final_activation=True,
        )
        self.head = PositivePrecipitationHead(
            in_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.output_channels,
            precipitation_channel=config.precipitation_channel,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.backbone(x))
        channels = list(torch.unbind(out, dim=-1))
        if self.config.use_coarse_temperature_residual and self.config.output_channels >= 1:
            channels[0] = channels[0] + x[..., self.config.coarse_temperature_channel]
        if self.config.use_coarse_precipitation_residual and self.config.output_channels >= 2:
            channels[1] = channels[1] + x[..., self.config.coarse_precipitation_channel]
        if self.config.use_coarse_wind_residual and self.config.output_channels >= 4:
            channels[2] = channels[2] + x[..., self.config.coarse_u_channel]
            channels[3] = channels[3] + x[..., self.config.coarse_v_channel]
        return torch.stack(channels, dim=-1)


def build_mlp_baseline_from_config(config: dict | None = None) -> NodewiseMLPBaseline:
    config = config or {}
    allowed = set(MLPBaselineConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in config.items() if key in allowed}
    return NodewiseMLPBaseline(MLPBaselineConfig(**kwargs))


def demo() -> None:
    model = NodewiseMLPBaseline()
    x = torch.randn(2, 8, 11)
    y = model(x)
    assert y.shape == (2, 8, 4)


if __name__ == "__main__":
    demo()
