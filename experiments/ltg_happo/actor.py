"""HAPPO actor adapter for the LTG trajectory/graph encoder."""

from __future__ import annotations

import torch
from torch import nn

from algorithms.actor_critic import Actor as HAPPOActor
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from experiments.ltg_happo.model import LTGActorCore


class PackedLTGEncoder(nn.Module):
    """Unpack causal per-agent histories and shared Sundial paths.

    The runner stores all three agents' observed histories, the frozen
    Sundial scenarios, and a one-hot agent identity in each replay observation.
    The actor forms masked messages only from adjacent supply-chain stages.
    """

    def __init__(self, args):
        super().__init__()
        self.agent_count = int(args.num_agents)
        self.history_window = int(args.ltg_history_window)
        self.obs_dim = int(args.ltg_obs_dim)
        self.path_count = int(args.ltg_path_count)
        self.forecast_horizon = int(args.ltg_forecast_horizon)
        self.hidden_dim = int(args.hidden_size)
        self.history_size = self.agent_count * self.history_window * self.obs_dim
        self.forecast_size = self.path_count * self.forecast_horizon
        self.expected_input_size = self.history_size + self.forecast_size + self.agent_count
        self.core = LTGActorCore(
            observation_dim=self.obs_dim,
            forecast_horizon=self.forecast_horizon,
            hidden_dim=self.hidden_dim,
            num_regimes=4,
            lead_time_buckets=16,
        )

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 2 or packed.shape[-1] != self.expected_input_size:
            raise ValueError(
                f"LTG packed observation shape must be [batch, {self.expected_input_size}]"
            )
        batch = packed.shape[0]
        histories = packed[:, : self.history_size].reshape(
            batch, self.agent_count, self.history_window, self.obs_dim
        )
        start = self.history_size
        paths = packed[:, start : start + self.forecast_size].reshape(
            batch, self.path_count, self.forecast_horizon
        )
        agent_one_hot = packed[:, start + self.forecast_size :]
        agent_ids = agent_one_hot.argmax(dim=-1)
        batch_ids = torch.arange(batch, device=packed.device)
        local_history = histories[batch_ids, agent_ids]
        forecast_embedding = self.core.forecast_encoder(paths, batch, local_history[:, 0, :])

        neighbor_messages = []
        neighbor_validity = []
        for offset in (-1, 1):
            neighbor_ids = agent_ids + offset
            valid = (neighbor_ids >= 0) & (neighbor_ids < self.agent_count)
            safe_ids = neighbor_ids.clamp(0, self.agent_count - 1)
            neighbor_history = histories[batch_ids, safe_ids]
            message = self.core.encode_message(neighbor_history, paths, forecast_embedding)
            neighbor_messages.append(message)
            neighbor_validity.append(valid)

        incoming = torch.stack(neighbor_messages, dim=1)
        edge_mask = torch.stack(neighbor_validity, dim=1)
        lead_times = torch.full(
            (batch, 2), 4, dtype=torch.long, device=packed.device
        )
        output = self.core(
            local_history=local_history,
            sundial_paths=paths,
            incoming_messages=incoming,
            lead_times=lead_times,
            edge_mask=edge_mask,
            forecast_embedding=forecast_embedding,
        )
        return output.actor_features


class LTGHAPPOActor(HAPPOActor):
    """Keep HAPPO's RNN and action distribution; replace only its input encoder."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        nn.Module.__init__(self)
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = PackedLTGEncoder(args).to(device)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        self.to(device)
