# Retained trained policies

This directory preserves Maskable PPO policy archives associated with the Transport Policy study and its reviewer-response analyses. These files support policy inspection and re-evaluation without requiring retraining.

## Collections

- `Results_eval/` contains the principal evaluation policy set: the DB Approach 1 policy and its lower/upper-bound variants, plus one forecast-based policy for each day in the 209-day evaluation set.
- `Results_2/` contains the severe-day/curriculum policy set used during focused analysis of the most congested days, including lower-bound variants.
- `ablation/` contains the retained policies for the reviewer-response ablation and sensitivity experiments, including direct-separation, no-curriculum, and separation-cap variants.

The exact filenames are historical artifacts and are intentionally preserved even where naming is terse.

## Software compatibility

The prepared experiment environment records Stable-Baselines3 2.4.0, SB3-Contrib 2.4.0, Gymnasium 1.0.0, PyTorch 2.6.0, and NumPy 1.26.4. See `../requirements.txt` for the full environment specification used by this repository.

## Reproducibility boundary

Retained policy archives allow the corresponding policies to be loaded and evaluated without retraining. They do **not** establish deterministic retraining: the original RL training was stochastic, and the public package does not claim that a new training run will recreate the same model bytes or numerical returns exactly.

CI checks the expected inventory (209 forecast-based policies under `Results_eval/` and 45 ablation policies), pins principal policy archives by exact Git blob ID, and validates representative archives as readable ZIP containers. CI intentionally does not execute the complete 209-policy evaluation because that is a substantially heavier computational workflow.

See [`../PROVENANCE.md`](../PROVENANCE.md) for the complete provenance statement.
