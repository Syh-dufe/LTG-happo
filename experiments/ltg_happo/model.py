"""Core decentralized encoder for the experimental LTG-HAPPO actor.

The module deliberately separates local encoding from message aggregation:
each agent can encode only its own observed history and forecast paths, emit a
message, receive messages from permitted neighbors, and then form actor
features. HAPPO's existing ACTLayer and centralized Critic can consume those
features after the rollout/buffer adapter is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LTGActorOutput:
    """Features and diagnostics produced for one decentralized actor."""

    actor_features: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    outgoing_message: Tensor
    forecast_embedding: Tensor
    neighbor_context: Tensor


class _SequenceEncoder(nn.Module):
    """Encode a batch of variable-length-in-value, fixed-window sequences."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        encoded, _ = self.gru(self.input_projection(sequence))
        return encoded[:, -1]


class _ForecastScenarioEncoder(nn.Module):
    """Encode a path distribution without expanding every path recurrently.

    All generated scenarios are retained through per-horizon mean, dispersion,
    and lower/upper quartile summaries. This keeps distributional information
    while making PPO minibatch evaluation tractable on CPU.
    """

    def __init__(self, horizon: int, hidden_dim: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.aggregate = nn.Sequential(
            nn.Linear(4 * horizon, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, paths: Optional[Tensor], batch_size: int, reference: Tensor) -> Tensor:
        if paths is None:
            return reference.new_zeros((batch_size, self.aggregate[0].out_features))
        if paths.ndim != 3 or paths.shape[0] != batch_size:
            raise ValueError("forecast paths must have shape [batch, scenarios, horizon]")
        if paths.shape[-1] != self.horizon:
            raise ValueError(
                f"forecast horizon is {paths.shape[-1]}, expected {self.horizon}"
            )
        if paths.shape[1] < 1:
            raise ValueError("at least one forecast scenario is required")

        if paths.shape[-1] != self.horizon:
            raise ValueError(f"forecast horizon is {paths.shape[-1]}, expected {self.horizon}")
        mean = paths.mean(dim=1)
        dispersion = paths.var(dim=1, unbiased=False).clamp_min(1e-8).sqrt()
        lower = torch.quantile(paths, 0.25, dim=1)
        upper = torch.quantile(paths, 0.75, dim=1)
        return self.aggregate(torch.cat((mean, dispersion, lower, upper), dim=-1))


class _LeadTimeMessageAttention(nn.Module):
    """Aggregate only supplied neighbor messages, conditioned on edge lead time."""

    def __init__(self, hidden_dim: int, lead_time_buckets: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lead_time_embedding = nn.Embedding(lead_time_buckets, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        local_context: Tensor,
        incoming_messages: Optional[Tensor],
        lead_times: Optional[Tensor],
        edge_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size = local_context.shape[0]
        if incoming_messages is None:
            return local_context.new_zeros((batch_size, self.hidden_dim))
        if incoming_messages.ndim != 3:
            raise ValueError("incoming_messages must have shape [batch, neighbors, hidden]")
        if incoming_messages.shape[0] != batch_size or incoming_messages.shape[-1] != self.hidden_dim:
            raise ValueError("incoming message dimensions do not match local context")

        neighbor_count = incoming_messages.shape[1]
        if neighbor_count == 0:
            return local_context.new_zeros((batch_size, self.hidden_dim))
        if lead_times is None or lead_times.shape != (batch_size, neighbor_count):
            raise ValueError("lead_times must have shape [batch, neighbors]")
        if edge_mask is None:
            edge_mask = torch.ones_like(lead_times, dtype=torch.bool)
        elif edge_mask.shape != lead_times.shape:
            raise ValueError("edge_mask must have shape [batch, neighbors]")
        else:
            edge_mask = edge_mask.to(dtype=torch.bool)

        bucket_count = self.lead_time_embedding.num_embeddings
        lead_buckets = lead_times.long().clamp(min=0, max=bucket_count - 1)
        edge_context = incoming_messages + self.lead_time_embedding(lead_buckets)
        query = self.query(local_context).unsqueeze(1)
        keys = self.key(edge_context)
        values = self.value(edge_context)
        scores = (query * keys).sum(dim=-1) / (self.hidden_dim**0.5)
        scores = scores.masked_fill(~edge_mask, torch.finfo(scores.dtype).min)

        # An agent can have no permitted incoming edges. Avoid softmax over an
        # all-masked row and return a zero context for that case.
        has_neighbor = edge_mask.any(dim=1, keepdim=True)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(has_neighbor, weights, torch.zeros_like(weights))
        aggregate = (weights.unsqueeze(-1) * values).sum(dim=1)
        return self.output(aggregate) * has_neighbor.to(aggregate.dtype)


class LTGActorCore(nn.Module):
    """Local history + Sundial forecast + lead-time messages -> HAPPO features.

    Input tensors are kept explicit so rollout code can enforce causal access:

    * ``local_history``: ``[B, T, D]`` observations available to this agent;
    * ``sundial_paths``: ``[B, S, H]`` forecast scenarios, never realized futures;
    * ``incoming_messages``: ``[B, M, hidden_dim]`` messages from allowed peers;
    * ``lead_times`` and ``edge_mask``: edge metadata for those peers.

    ``actor_features`` can be passed to the existing HAPPO ``ACTLayer``. The
    auxiliary regime head is trained only with labels made from training
    trajectories; the module does not infer labels from future test outcomes.
    """

    def __init__(
        self,
        observation_dim: int,
        forecast_horizon: int,
        hidden_dim: int,
        num_regimes: int = 4,
        lead_time_buckets: int = 16,
    ) -> None:
        super().__init__()
        if min(observation_dim, forecast_horizon, hidden_dim, num_regimes) < 1:
            raise ValueError("all model dimensions must be positive")
        self.hidden_dim = hidden_dim
        self.history_encoder = _SequenceEncoder(observation_dim, hidden_dim)
        self.forecast_encoder = _ForecastScenarioEncoder(forecast_horizon, hidden_dim)
        self.regime_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_regimes),
        )
        self.regime_embeddings = nn.Embedding(num_regimes, hidden_dim)
        self.message_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_attention = _LeadTimeMessageAttention(hidden_dim, lead_time_buckets)
        self.actor_fusion = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def encode_message(
        self,
        local_history: Tensor,
        sundial_paths: Optional[Tensor],
        forecast_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a neighbor's causal history and forecast as a graph message.

        The receiving actor applies its own message encoder to the neighbor's
        public local trajectory.  This keeps the original decentralized
        information boundary while allowing each HAPPO actor to learn which
        neighbor signals matter for its ordering decision.
        """
        history = self.history_encoder(local_history)
        forecast = forecast_embedding if forecast_embedding is not None else self.forecast_encoder(
            sundial_paths, local_history.shape[0], history
        )
        regime_logits = self.regime_head(torch.cat((history, forecast), dim=-1))
        regime_probabilities = torch.softmax(regime_logits, dim=-1)
        regime_context = regime_probabilities @ self.regime_embeddings.weight
        return self.message_projection(torch.cat((history, forecast, regime_context), dim=-1))

    def forward(
        self,
        local_history: Tensor,
        sundial_paths: Optional[Tensor] = None,
        incoming_messages: Optional[Tensor] = None,
        lead_times: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
        forecast_embedding: Optional[Tensor] = None,
    ) -> LTGActorOutput:
        if local_history.ndim != 3:
            raise ValueError("local_history must have shape [batch, time, features]")
        history = self.history_encoder(local_history)
        forecast = forecast_embedding if forecast_embedding is not None else self.forecast_encoder(
            sundial_paths, local_history.shape[0], history
        )

        regime_logits = self.regime_head(torch.cat((history, forecast), dim=-1))
        regime_probabilities = torch.softmax(regime_logits, dim=-1)
        regime_context = regime_probabilities @ self.regime_embeddings.weight

        local_context = torch.cat((history, forecast, regime_context), dim=-1)
        outgoing_message = self.message_projection(local_context)
        neighbor_context = self.message_attention(
            local_context=history + forecast + regime_context,
            incoming_messages=incoming_messages,
            lead_times=lead_times,
            edge_mask=edge_mask,
        )
        actor_features = self.actor_fusion(
            torch.cat((history, forecast, regime_context, neighbor_context), dim=-1)
        )
        return LTGActorOutput(
            actor_features=actor_features,
            regime_logits=regime_logits,
            regime_probabilities=regime_probabilities,
            outgoing_message=outgoing_message,
            forecast_embedding=forecast,
            neighbor_context=neighbor_context,
        )

    @staticmethod
    def regime_auxiliary_loss(
        regime_logits: Tensor,
        regime_labels: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Cross-entropy for training-only trajectory regime labels."""
        labels = regime_labels.to(device=regime_logits.device, dtype=torch.long).reshape(-1)
        logits = regime_logits.reshape(-1, regime_logits.shape[-1])
        if labels.numel() != logits.shape[0]:
            raise ValueError("regime label count must match the logits batch")
        if valid_mask is None:
            return F.cross_entropy(logits, labels)
        mask = valid_mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != labels.numel():
            raise ValueError("valid_mask size must match the logits batch")
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask], labels[mask])
