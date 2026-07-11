"""Physics-informed graph neural network for weather downscaling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .layers import MLP, PositivePrecipitationHead, ResidualGraphBlock


@dataclass(frozen=True)
class PIGNNConfig:
    """Configuration for :class:`PIGNN`.

    The default input layout matches ``WeatherGraphDataset``:
    five dynamic atmospheric channels, three static terrain channels, and two
    temporal channels.
    """

    node_input_channels: int = 10
    edge_input_channels: int = 3
    hidden_channels: int = 128
    edge_hidden_channels: int = 32
    message_hidden_channels: int = 128
    output_channels: int = 2
    num_layers: int = 4
    dropout: float = 0.1
    precipitation_channel: int | None = 1
    use_coarse_temperature_residual: bool = True
    use_coarse_precipitation_residual: bool = False


class PIGNN(nn.Module):
    """Static-topology, dynamic-node-feature PI-GNN.

    Inputs:

    * ``x``: ``[N, 10]`` or ``[B, N, 10]`` node features.
    * ``edge_index``: ``[2, E]`` fixed directed k-NN graph.
    * ``edge_attr``: ``[E, 3]`` fixed edge attributes.

    Outputs:

    * ``[N, 2]`` or ``[B, N, 2]`` predictions. By convention channel 0 is
      temperature and channel 1 is precipitation.

    The model is not time-recurrent. Each timestep is a sample on the same
    static graph. Temporal context can later be added by feeding lagged dynamic
    variables or wrapping this model in a sequence model.
    """

    def __init__(self, config: PIGNNConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = PIGNNConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config

        self.node_encoder = MLP(
            in_channels=config.node_input_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.hidden_channels,
            num_layers=2,
            dropout=config.dropout,
            final_activation=True,
        )
        self.edge_encoder = MLP(
            in_channels=config.edge_input_channels,
            hidden_channels=config.edge_hidden_channels,
            out_channels=config.edge_hidden_channels,
            num_layers=2,
            dropout=config.dropout,
            final_activation=True,
        )
        self.blocks = nn.ModuleList(
            [
                ResidualGraphBlock(
                    hidden_channels=config.hidden_channels,
                    edge_channels=config.edge_hidden_channels,
                    message_hidden_channels=config.message_hidden_channels,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_channels)
        self.head = PositivePrecipitationHead(
            in_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.output_channels,
            precipitation_channel=config.precipitation_channel,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        encoded_edges = self.edge_encoder(edge_attr)

        h = self.node_encoder(x)
        for block in self.blocks:
            h = block(h, edge_index.to(x.device), encoded_edges)

        out = self.head(self.norm(h))
        return self._add_coarse_residuals(out, x)

    def _add_coarse_residuals(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Optionally predict corrections around coarse ERA5 channels.

        Temperature downscaling often benefits from residual learning:
        prediction = coarse t2m + learned correction. Precipitation targets may
        or may not use an ERA5 coarse precipitation channel; this project does
        not currently include one, so precipitation residuals are disabled by
        default.
        """

        channels = list(torch.unbind(out, dim=-1))
        if self.config.use_coarse_temperature_residual and self.config.output_channels >= 1:
            channels[0] = channels[0] + x[..., 0]
        if self.config.use_coarse_precipitation_residual and self.config.output_channels >= 2:
            channels[1] = channels[1] + x[..., 1]
        return torch.stack(channels, dim=-1)


def build_pignn_from_config(config: dict | None = None) -> PIGNN:
    """Build a ``PIGNN`` from a plain dictionary section."""

    config = config or {}
    allowed = set(PIGNNConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in config.items() if key in allowed}
    return PIGNN(PIGNNConfig(**kwargs))
