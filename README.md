# LTG-HAPPO

Lead-Time-aware Trajectory-Graph HAPPO (LTG-HAPPO) is an experimental actor architecture for multi-echelon inventory control. It keeps the HAPPO training objective, sequential multi-agent update, recurrent policy head, action distribution, and centralized critic. It replaces the actor's input encoder with a learned representation of local history, frozen Sundial demand scenarios, and lead-time-aware messages from adjacent echelons.

## Architecture

```text
Observed local history ──> GRU history encoder ───────────┐
                                                          ├─> regime head / regime embedding ─┐
32 frozen Sundial paths ─> mean, std, Q25, Q75 by horizon ─┘                                    │
                                                                                                ├─> actor fusion ─> HAPPO RNN ─> action distribution
Adjacent-echelon histories ─> messages + lead-time attention ──────────────────────────────────┘

Original centralized critic and HAPPO sequential policy updates remain in place.
```

The Sundial model is frozen and used for zero-shot inference. Its sampled paths are compressed into four statistics at each forecast horizon before entering the actor. The graph encoder only accepts the supplied neighboring messages and masks unavailable edges. In the current three-echelon setting, the edge lead time is four periods. The code does not fine-tune Sundial or include its model weights.

## Source layout

- `experiments/ltg_happo/model.py`: GRU history encoder, forecast-distribution encoder, regime representation, lead-time-aware message attention, and actor fusion.
- `experiments/ltg_happo/actor.py`: adapter that feeds the LTG features into HAPPO's existing recurrent layer and action head.
- `experiments/ltg_happo/runner.py`: rollout adapter that stores causal histories, requests Sundial forecasts, and packs observations for the actor.

## Integration requirements

This repository contains the LTG-HAPPO architecture files, not a standalone copy of the upstream HAPPO project or the Sundial model. Integrate these files into a compatible HAPPO checkout that provides `algorithms.actor_critic`, `algorithms.utils.rnn`, `algorithms.utils.act`, and `runners.separated.runner`.

The runner expects a predictor module at `experiments.sundial_closed_loop_development.predictor` with `model()` and `parameter_hash()` functions. Register the expected model hash in the run manifest. The predictor must use only demand observations available up to the current period and return finite paths shaped `[batch, scenarios, horizon]`. The environment wrapper must expose `signal_obs_dim`; the original centralized observations are retained for the critic.

The implementation was developed against Liu et al.'s HAPPO inventory code at commit `a7e5a3e83e21565a5799483bc534e39635ec65dd`. Environment-specific training configuration and local model-loading scripts are intentionally not included because they contain experiment and machine setup details. No performance claim follows from this architecture source alone; compare policies on held-out demand paths before making such a claim.

## Dependencies

The architecture modules require Python, PyTorch, and NumPy. HAPPO and Sundial dependencies are supplied by the host project as described above.

## Data

The paired seed-level evaluation demand inputs and the training-demand generation protocol are in [data/README.md](data/README.md). The protocol records that historical online training draws were not archived.
