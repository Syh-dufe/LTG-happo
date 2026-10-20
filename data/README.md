# Data package

## Evaluation demand traces

`test_demand/seed_11.csv` through `seed_15.csv` contain the external demand observed by the downstream node for the paired evaluation of model seeds 11–15. Each file has the same 80 paths (40 `base`, 40 `shock`) and 200 periods per path. The five files repeat identical demand sequences and label them by compared model seed, so the model seeds are evaluated on paired inputs. The columns identify the model seed, path, scenario, batch, trace, period, and realized external demand. `test_dataset_manifest.json` records path counts and SHA-256 checksums.

These inputs were extracted from the saved HAPPO evaluation records and reused in later LTG-HAPPO diagnostics. They are useful for paired comparisons, but they are not a fresh, untouched holdout. Do not describe them as independent test data in a paper.

## Training demand

The HAPPO/LTG-HAPPO training environment generates demand online at episode reset; it does not load a fixed training CSV. The realized per-episode training trajectories were not archived, so this repository does not claim to contain those historical draws. The training protocol for seeds 11–15 is recorded in `merton_training_protocol.json`: it specifies the upstream Merton generator and seed IDs, but is not a substitute for the missing exact trajectories.

To reproduce the training distribution, use the HAPPO environment generator at the source revision recorded there. Each environment reset samples fresh Merton demand paths; the seed initializes the run's random generators. Since worker-level random draws were not saved, exact bit-for-bit reconstruction of the historical training stream cannot be guaranteed.
