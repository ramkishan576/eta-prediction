# CLAUDE.md — AI Tooling Log

## Tool used
Claude Sonnet (claude.ai) for architecture design and code generation.

## How it was used

### Architecture decisions
Prompted Claude to analyse the assignment constraints (MAE metric, Dockerized
submission, no external API calls at inference, 2024 eval slice) and propose
a feature set.  The model recommended LightGBM with an `regression_l1`
objective to directly optimise MAE, which I adopted.

### Feature engineering
Discussed what signals matter for NYC taxi duration:
- Great-circle and Manhattan distance (primary drivers).
- Bearing — directional asymmetry in NYC traffic (e.g. going uptown vs
  downtown in rush hour differs).
- Cyclic hour/day-of-week encodings to prevent ordinal distance artefacts.
- K-Means location clusters (100 clusters on all pickup + dropoff coordinates)
  to capture neighbourhood-level demand patterns.
- Airport proximity flags because JFK/LGA/EWR trips have systematically
  different duration distributions.

### Code generation workflow
1. Wrote `src/config.py` first to centralise all constants and paths.
2. Generated `src/features.py` as a stateless transformation module.
3. Generated `train.py` and `predict.py` against the feature module interface.
4. Reviewed generated code manually — removed redundant comments, validated
   bounding box constants against NYC geography, confirmed the K-Means
   re-hydration pattern in `predict.py` (injecting `cluster_centers_` without
   re-fitting).

### What I changed manually
- Tightened the NYC bounding box constants after verifying against TLC data
  documentation (some raw Kaggle rows contain coordinates in New Jersey or
  outside the five boroughs).
- Replaced an initial `StandardScaler` wrapping the target with direct MAE
  optimisation — LightGBM's `regression_l1` objective handles this natively.
- Added the `is_night` / `is_am_rush` / `is_pm_rush` binary flags after
  reviewing feature importance from an early run where `pickup_hour` alone
  ranked very high, suggesting non-linear hour effects.

## What AI did not decide
- Dataset choice (Kaggle NYC Taxi Trip Duration 2016 train.csv).
- Whether to train on log(duration) vs raw duration — I tested both; raw MAE
  optimisation via `regression_l1` outperformed log-transform + MSE on this
  metric by ~8s on the validation fold.
- Hyperparameter values for `num_leaves` and `feature_fraction` — tuned
  manually on a 10 % sample before running full training.
