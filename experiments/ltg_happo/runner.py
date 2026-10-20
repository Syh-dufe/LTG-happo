"""HAPPO rollout adapter that stores causal history and frozen Sundial paths."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from runners.separated.runner import CRunner


class LTGRunner(CRunner):
    def __init__(self, config):
        args = config["all_args"]
        envs = config["envs"]
        self.agent_count = int(config["num_agents"])
        self.raw_obs_dim = int(envs.signal_obs_dim)
        self.history_window = int(args.ltg_history_window)
        self.path_count = int(args.ltg_path_count)
        self.forecast_horizon = int(args.ltg_forecast_horizon)
        self.forecast_refresh = int(args.ltg_forecast_refresh)
        self.sundial_context = int(args.ltg_sundial_context)
        self.lead_time = int(args.ltg_lead_time)
        args.ltg_obs_dim = self.raw_obs_dim
        packed_dim = (
            self.agent_count * self.history_window * self.raw_obs_dim
            + self.path_count * self.forecast_horizon
            + self.agent_count
        )
        self._set_packed_observation_spaces(envs, packed_dim)
        if config.get("eval_envs") is not None:
            self._set_packed_observation_spaces(config["eval_envs"], packed_dim)
        super().__init__(config)

        self.local_histories = []
        self.demand_histories = []
        self.forecast_paths = None
        self.episode_step = 0
        self.current_rollout = 0
        self.sundial_calls = 0
        self.sundial_seconds = 0.0
        self._predictor = None
        self._forecast_model = None
        self._progress_file = Path(self.run_dir).parent / "progress.json"
        self._curve_file = Path(self.run_dir).parent / "curve.csv"
        self._eval_curve_file = Path(self.run_dir).parent / "validation_curve.csv"
        self._manifest = self._load_training_manifest()
        self.eval_interval = int(args.eval_interval)
        self.eval_patience = int(args.n_no_improvement_thres)
        self.eval_warmup = int(args.n_warmup_evaluations)
        self.best_validation_cost = float("inf")
        self.no_improvement_evaluations = 0
        self.best_validation_rollout = None
        self.validation_evaluations = 0
        self.eval_sundial_calls = 0
        self.eval_sundial_seconds = 0.0
        self._in_evaluation = False
        self._validation_trace_index = 0
        self._eval_forecast_cache = {}

    @staticmethod
    def _set_packed_observation_spaces(envs, packed_dim):
        envs.observation_space = [
            space.__class__(low=-np.inf, high=np.inf, shape=(packed_dim,), dtype=np.float32)
            for space in envs.observation_space
        ]

    def _load_training_manifest(self):
        path = Path(self.run_dir).parent / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Training registration missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_progress(self, status, episode, total_steps, train_cost=None):
        payload = {
            "status": status,
            "seed": int(self.all_args.seed),
            "completed_rollouts": int(episode),
            "completed_env_steps": int(total_steps),
            "sampled_env_steps": int(
                total_steps
                + (self.episode_step if episode <= self.current_rollout else 0) * self.n_rollout_threads
            ),
            "current_rollout_periods": int(self.episode_step),
            "requested_env_steps": int(self.num_env_steps),
            "forecast_refresh_periods": self.forecast_refresh,
            "sundial_inference_calls": self.sundial_calls,
            "sundial_inference_seconds": round(self.sundial_seconds, 3),
            "last_training_cost": None if train_cost is None else float(train_cost),
            "best_validation_cost": None if not np.isfinite(self.best_validation_cost) else float(self.best_validation_cost),
            "no_improvement_evaluations": int(self.no_improvement_evaluations),
            "validation_evaluations": int(self.validation_evaluations),
            "updated_unix": time.time(),
        }
        temp = self._progress_file.with_name(
            f"{self._progress_file.stem}.{time.time_ns()}.tmp"
        )
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(10):
            try:
                temp.replace(self._progress_file)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def _load_predictor(self):
        if self._predictor is None:
            from experiments.sundial_closed_loop_development import predictor

            self._predictor = predictor
            self._forecast_model = predictor.model()
            actual_hash = predictor.parameter_hash()
            expected_hash = self._manifest["sundial_parameter_sha256"]
            if actual_hash != expected_hash:
                raise RuntimeError("Frozen Sundial parameter hash differs from registration")
        return self._forecast_model

    def _refresh_sundial_paths(self):
        """Forecast root external demand using history available at this step only."""
        histories = self.demand_histories
        if not histories or not all(histories):
            return
        cache_key = None
        if self._in_evaluation:
            cache_key = (self._validation_trace_index, self.episode_step)
            cached = self._eval_forecast_cache.get(cache_key)
            if cached is not None:
                self.forecast_paths = cached.copy()
                return
        model = self._load_predictor()
        series = []
        for values in histories:
            recent = np.asarray(values[-self.sundial_context :], dtype=np.float32)
            if recent.size < self.sundial_context:
                recent = np.pad(
                    recent,
                    (self.sundial_context - recent.size, 0),
                    mode="edge",
                )
            series.append(recent)
        model_device = next(model.parameters()).device
        inputs = torch.as_tensor(np.stack(series), dtype=torch.float32, device=model_device)
        old_rng_state = torch.random.get_rng_state()
        old_cuda_rng_state = (
            torch.cuda.get_rng_state(model_device) if model_device.type == "cuda" else None
        )
        seed_material = f"ltg-happo-v1|{self.all_args.seed}|{self.current_rollout}|{self.episode_step}"
        seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big") % (2**63 - 1)
        torch.manual_seed(seed)
        started = time.perf_counter()
        try:
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            with torch.inference_mode():
                paths = model.generate(
                    inputs,
                    max_new_tokens=self.forecast_horizon,
                    num_samples=self.path_count,
                    revin=True,
                )
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
        finally:
            torch.random.set_rng_state(old_rng_state)
            if old_cuda_rng_state is not None:
                torch.cuda.set_rng_state(old_cuda_rng_state, model_device)
        elapsed = time.perf_counter() - started
        paths = paths.detach().cpu().numpy()
        expected = (len(histories), self.path_count, self.forecast_horizon)
        if paths.shape != expected or not np.isfinite(paths).all():
            raise RuntimeError(f"Invalid Sundial output: expected {expected}, got {paths.shape}")
        # The inventory environment scales observations and demand to [0, 1].
        self.forecast_paths = np.clip(paths, 0.0, 20.0).astype(np.float32) / 20.0
        if self._in_evaluation:
            self.eval_sundial_calls += 1
            self.eval_sundial_seconds += elapsed
            self._eval_forecast_cache[cache_key] = self.forecast_paths.copy()
        else:
            self.sundial_calls += 1
            self.sundial_seconds += elapsed

    def _pack_observations(self, raw_obs):
        env_count = raw_obs.shape[0]
        packed = np.zeros(
            (env_count, self.agent_count, self.agent_count * self.history_window * self.raw_obs_dim
             + self.path_count * self.forecast_horizon + self.agent_count),
            dtype=np.float32,
        )
        histories = np.zeros(
            (env_count, self.agent_count, self.history_window, self.raw_obs_dim),
            dtype=np.float32,
        )
        for env_id, sequence in enumerate(self.local_histories):
            available = np.asarray(sequence[-self.history_window :], dtype=np.float32)
            histories[env_id, :, -len(available) :] = available.transpose(1, 0, 2)
        history_size = self.agent_count * self.history_window * self.raw_obs_dim
        forecast_size = self.path_count * self.forecast_horizon
        for agent_id in range(self.agent_count):
            packed[:, agent_id, :history_size] = histories.reshape(env_count, -1)
            packed[:, agent_id, history_size : history_size + forecast_size] = self.forecast_paths.reshape(env_count, -1)
            packed[:, agent_id, history_size + forecast_size + agent_id] = 1.0
        return packed

    def warmup(self):
        raw_obs, available_actions = self.envs.reset()
        self.local_histories = [[raw_obs[i].copy()] for i in range(self.n_rollout_threads)]
        self.demand_histories = [[] for _ in range(self.n_rollout_threads)]
        self.forecast_paths = np.zeros(
            (self.n_rollout_threads, self.path_count, self.forecast_horizon), dtype=np.float32
        )
        self.episode_step = 0
        packed_obs = self._pack_observations(raw_obs)
        share_obs = np.asarray([np.concatenate(raw_obs[i], axis=0) for i in range(self.n_rollout_threads)])
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs
            self.buffer[agent_id].obs[0] = packed_obs[:, agent_id]
            self.buffer[agent_id].available_actions[0] = None

    def insert(self, data):
        (raw_obs, share_obs, rewards, dones, infos, available_actions, values, actions,
         action_log_probs, rnn_states, rnn_states_critic) = data
        self.episode_step += 1
        for env_id in range(self.n_rollout_threads):
            self.local_histories[env_id].append(raw_obs[env_id].copy())
            if len(self.local_histories[env_id]) > self.history_window:
                self.local_histories[env_id].pop(0)
            # The downstream agent observes the latest realized external demand.
            self.demand_histories[env_id].append(float(raw_obs[env_id, 0, 2]) * 20.0)
            if len(self.demand_histories[env_id]) > self.sundial_context:
                self.demand_histories[env_id].pop(0)
        if self.episode_step % self.forecast_refresh == 0:
            self._refresh_sundial_paths()
            self._write_progress(
                "training",
                self.current_rollout,
                self.current_rollout * self.episode_length * self.n_rollout_threads,
            )
        packed_obs = self._pack_observations(raw_obs)

        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env] = np.zeros(
            ((dones_env).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        rnn_states_critic[dones_env] = np.zeros(
            ((dones_env).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = 0.0
        for agent_id in range(self.num_agents):
            current_share_obs = share_obs
            if not self.use_centralized_V:
                current_share_obs = raw_obs[:, agent_id]
            self.buffer[agent_id].insert(
                current_share_obs,
                packed_obs[:, agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def evaluate_validation(self):
        """Evaluate deterministically on the fixed Merton paths used by HAPPO."""
        if self.eval_envs is None:
            raise RuntimeError("validation-based early stopping requires eval_envs")
        eval_count = int(self.eval_envs.get_eval_num())
        if eval_count != 20:
            raise RuntimeError(f"expected the registered 20 Merton validation paths, found {eval_count}")

        previous_rollout = self.current_rollout
        episode_costs = []
        self._in_evaluation = True
        try:
            for trace_index in range(eval_count):
                raw_obs, _ = self.eval_envs.reset()
                self._validation_trace_index = trace_index
                # Keep Sundial's stochastic samples identical across evaluations
                # of different checkpoints on the same fixed demand trace.
                self.current_rollout = -100000 - trace_index
                self.local_histories = [[raw_obs[0].copy()]]
                self.demand_histories = [[]]
                self.forecast_paths = np.zeros(
                    (1, self.path_count, self.forecast_horizon), dtype=np.float32
                )
                self.episode_step = 0
                rnn_states = np.zeros(
                    (1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
                )
                masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                period_costs = []

                for _ in range(self.episode_length):
                    packed_obs = self._pack_observations(raw_obs)
                    actions_env = []
                    next_states = []
                    for agent_id in range(self.num_agents):
                        self.trainer[agent_id].prep_rollout()
                        action, next_state = self.trainer[agent_id].policy.act(
                            packed_obs[:, agent_id],
                            rnn_states[:, agent_id],
                            masks[:, agent_id],
                            None,
                            deterministic=True,
                        )
                        next_states.append(next_state.detach().cpu().numpy())
                        action_index = action.detach().cpu().numpy().astype(int).reshape(-1)
                        actions_env.append(
                            np.eye(self.eval_envs.action_space[agent_id].n, dtype=np.float32)[action_index[0]]
                        )
                    rnn_states = np.stack(next_states, axis=1)
                    raw_obs, rewards, dones, _ = self.eval_envs.step([actions_env])
                    period_costs.append(-float(np.mean(rewards[0])))

                    self.episode_step += 1
                    self.local_histories[0].append(raw_obs[0].copy())
                    if len(self.local_histories[0]) > self.history_window:
                        self.local_histories[0].pop(0)
                    self.demand_histories[0].append(float(raw_obs[0, 0, 2]) * 20.0)
                    if len(self.demand_histories[0]) > self.sundial_context:
                        self.demand_histories[0].pop(0)
                    if self.episode_step % self.forecast_refresh == 0:
                        self._refresh_sundial_paths()
                    masks.fill(1.0)
                    if bool(np.all(dones[0])):
                        masks.fill(0.0)

                episode_costs.append(float(np.mean(period_costs)))
        finally:
            self._in_evaluation = False
            self.current_rollout = previous_rollout

        return float(np.mean(episode_costs)), episode_costs

    def _save_best_validation_model(self):
        old_save_dir = self.save_dir
        self.save_dir = str(Path(self.run_dir).parent / "best_validation_model")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.save()
        self.save_dir = old_save_dir

    def run(self):
        start = time.time()
        rollouts = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        completed_rollouts = 0
        stop_reason = "step_budget"
        self._write_progress("training", 0, 0)
        try:
            for episode in range(rollouts):
                if self.use_eval and episode % self.eval_interval == 0:
                    validation_cost, trace_costs = self.evaluate_validation()
                    self.validation_evaluations += 1
                    eval_steps = completed_rollouts * self.episode_length * self.n_rollout_threads
                    with self._eval_curve_file.open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        if stream.tell() == 0:
                            writer.writerow([
                                "rollout_before_training", "env_steps", "mean_validation_cost",
                                "trace_count", "trace_costs_json", "best_validation_cost",
                                "no_improvement_evaluations",
                            ])
                        writer.writerow([
                            episode, eval_steps, validation_cost, len(trace_costs),
                            json.dumps(trace_costs),
                            None if not np.isfinite(self.best_validation_cost) else self.best_validation_cost,
                            self.no_improvement_evaluations,
                        ])

                    improved = episode > 0 and validation_cost < self.best_validation_cost
                    if improved:
                        self.best_validation_cost = validation_cost
                        self.best_validation_rollout = episode
                        self.no_improvement_evaluations = 0
                        self._save_best_validation_model()
                    elif episode > self.eval_warmup:
                        self.no_improvement_evaluations += 1

                    print(
                        f"LTG_HAPPO_VALIDATION rollout={episode} steps={eval_steps} "
                        f"mean_cost={validation_cost:.4f} best={self.best_validation_cost:.4f} "
                        f"no_improve={self.no_improvement_evaluations}/{self.eval_patience} "
                        f"paths={len(trace_costs)} eval_sundial_calls={self.eval_sundial_calls}",
                        flush=True,
                    )
                    self._write_progress("validation", completed_rollouts, eval_steps)
                    if self.no_improvement_evaluations >= self.eval_patience:
                        stop_reason = "validation_patience"
                        break

                self.current_rollout = episode
                self.warmup()
                for step in range(self.episode_length):
                    values, actions, log_probs, actor_states, critic_states, env_actions = self.collect(step)
                    raw_obs, rewards, dones, infos = self.envs.step(env_actions)
                    share_obs = np.asarray([np.concatenate(raw_obs[i], axis=0) for i in range(self.n_rollout_threads)])
                    available_actions = None
                    data = (raw_obs, share_obs, rewards, dones, infos, available_actions,
                            values, actions, log_probs, actor_states, critic_states)
                    self.insert(data)
                train_cost = -float(np.mean([np.mean(buffer.rewards) for buffer in self.buffer]))
                self.compute()
                self.train()
                completed_rollouts = episode + 1
                total_steps = completed_rollouts * self.episode_length * self.n_rollout_threads
                with self._curve_file.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    if completed_rollouts == 1 and stream.tell() == 0:
                        writer.writerow(["rollout", "env_steps", "mean_training_cost", "sundial_calls"])
                    writer.writerow([episode + 1, total_steps, train_cost, self.sundial_calls])
                if completed_rollouts % 50 == 0 or completed_rollouts == rollouts:
                    old_save_dir = self.save_dir
                    self.save_dir = str(Path(self.run_dir).parent / "checkpoints" / f"step_{total_steps}")
                    Path(self.save_dir).mkdir(parents=True, exist_ok=True)
                    self.save()
                    self.save_dir = old_save_dir
                self._write_progress("training", completed_rollouts, total_steps, train_cost)
                if completed_rollouts % 5 == 0:
                    print(
                        f"LTG_HAPPO rollout={completed_rollouts}/{rollouts} steps={total_steps}/{self.num_env_steps} "
                        f"train_cost={train_cost:.4f} sundial_calls={self.sundial_calls} "
                        f"forecast_sec={self.sundial_seconds:.1f} elapsed_min={(time.time()-start)/60:.1f}",
                        flush=True,
                    )
            final_steps = completed_rollouts * self.episode_length * self.n_rollout_threads
            self.save_dir = str(Path(self.run_dir).parent / "final_models")
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            self.save()
            summary = {
                "status": "early_stopped" if stop_reason == "validation_patience" else "completed",
                "seed": int(self.all_args.seed),
                "requested_env_steps": int(self.num_env_steps),
                "completed_env_steps": final_steps,
                "rollouts": completed_rollouts,
                "stop_reason": stop_reason,
                "validation": {
                    "path_count": 20,
                    "evaluation_interval_rollouts": self.eval_interval,
                    "warmup_evaluations": self.eval_warmup,
                    "patience_evaluations": self.eval_patience,
                    "evaluation_count": self.validation_evaluations,
                    "best_mean_cost": None if not np.isfinite(self.best_validation_cost) else self.best_validation_cost,
                    "best_rollout_before_training": self.best_validation_rollout,
                    "best_model_directory": str(Path(self.run_dir).parent / "best_validation_model"),
                    "sundial_forecasts_cached_across_evaluations": True,
                    "evaluation_sundial_inference_calls": self.eval_sundial_calls,
                    "evaluation_sundial_inference_seconds": round(self.eval_sundial_seconds, 3),
                },
                "sundial_inference_calls": self.sundial_calls,
                "sundial_inference_seconds": round(self.sundial_seconds, 3),
                "training_wall_seconds": round(time.time() - start, 3),
                "evaluation": "validation-only Merton paths used for early stopping and checkpoint selection; heldout comparison remains separate",
                "forecast": "frozen zero-shot Sundial; 32 raw paths; refreshed every 10 periods; no adapter",
            }
            path = Path(self.run_dir).parent / "completed.json"
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_progress(stop_reason, completed_rollouts, final_steps)
            print("LTG_HAPPO_TRAINING_COMPLETED " + json.dumps(summary, ensure_ascii=False), flush=True)
            return summary
        except Exception as exc:
            path = Path(self.run_dir).parent / "failed.json"
            path.write_text(json.dumps({"error": repr(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
            completed_steps = self.current_rollout * self.episode_length * self.n_rollout_threads
            self._write_progress("failed", self.current_rollout, completed_steps)
            raise
