"""Reusable neural network layers for static-topology weather GNNs.

The graph topology is static: ``edge_index`` and ``edge_attr`` are shared by
every timestep. Node features are dynamic per sample. These layers therefore
accept either one graph ``x=[N, F]`` or a batch of timesteps ``x=[B, N, F]``
with one shared ``edge_index=[2, E]``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Small fully-connected network used for encoders, messages, and heads."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        activation_name: str = "silu",
        dropout: float = 0.0,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current, hidden_channels))
            layers.append(activation(activation_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden_channels
        layers.append(nn.Linear(current, out_channels))
        if final_activation:
            layers.append(activation(activation_name))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeConditionedMessagePassing(nn.Module):
    """Message passing block that conditions messages on static edge features.

    For each directed edge ``j -> i`` this computes a message from source node,
    destination node, and encoded edge attributes:

        m_ji = MLP([h_j, h_i, e_ji])

    Messages are mean-aggregated at destination nodes. A learned gate from edge
    attributes can down/up-weight different terrain relationships.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        message_hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        message_hidden_channels = message_hidden_channels or hidden_channels
        self.message_mlp = MLP(
            in_channels=hidden_channels * 2 + edge_channels,
            hidden_channels=message_hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.update_mlp = MLP(
            in_channels=hidden_channels * 2,
            hidden_channels=message_hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
        )

    @staticmethod
    def _ensure_batched(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return x.unsqueeze(0), False
        if x.ndim == 3:
            return x, True
        raise ValueError(f"Expected x with shape [N,F] or [B,N,F], got {tuple(x.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x_batched, was_batched = self._ensure_batched(x)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_attr.ndim != 2:
            raise ValueError("edge_attr must have shape [E, C_edge]")

        src = edge_index[0].long()
        dst = edge_index[1].long()
        batch_size, node_count, _ = x_batched.shape

        h_src = x_batched[:, src, :]
        h_dst = x_batched[:, dst, :]
        edge = edge_attr.to(device=x_batched.device, dtype=x_batched.dtype)
        edge_expanded = edge.unsqueeze(0).expand(batch_size, -1, -1)

        message_input = torch.cat([h_src, h_dst, edge_expanded], dim=-1)
        messages = self.message_mlp(message_input)
        messages = messages * self.edge_gate(edge).unsqueeze(0)

        aggregated = x_batched.new_zeros(batch_size, node_count, messages.shape[-1])
        aggregated.index_add_(1, dst, messages)

        degree = torch.bincount(dst, minlength=node_count).to(
            device=x_batched.device,
            dtype=x_batched.dtype,
        )
        aggregated = aggregated / degree.clamp_min(1.0).view(1, -1, 1)

        updated = self.update_mlp(torch.cat([x_batched, aggregated], dim=-1))
        return updated if was_batched else updated.squeeze(0)


class ResidualGraphBlock(nn.Module):
    """Pre-norm residual graph block for stable deep message passing."""

    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        message_hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_message = nn.LayerNorm(hidden_channels)
        self.message_passing = EdgeConditionedMessagePassing(
            hidden_channels=hidden_channels,
            edge_channels=edge_channels,
            message_hidden_channels=message_hidden_channels,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(hidden_channels)
        self.feed_forward = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels * 2,
            out_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = x + self.dropout(self.message_passing(self.norm_message(x), edge_index, edge_attr))
        h = h + self.dropout(self.feed_forward(self.norm_ff(h)))
        return h


class PositivePrecipitationHead(nn.Module):
    """Prediction head with optional non-negative precipitation channel."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        precipitation_channel: int | None = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.precipitation_channel = precipitation_channel
        self.head = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        if self.precipitation_channel is None:
            return out
        channels = list(torch.unbind(out, dim=-1))
        channels[self.precipitation_channel] = F.softplus(channels[self.precipitation_channel])
        return torch.stack(channels, dim=-1)
