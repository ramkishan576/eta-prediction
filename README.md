# NYC Taxi ETA — Gobblecube Hiring Challenge

Predicts NYC taxi trip duration (seconds) from pickup/dropoff coordinates and
datetime.  Submitted for the **ETA Challenge** track.

---

## Results

| Split | MAE (seconds) | Notes |
|---|---|---|
| Validation (15 % hold-out) | **~290s** | LightGBM, 3000 estimators |
| Repo baseline | 367s | — |

---

## Dataset

**Kaggle NYC Taxi Trip Duration (2016)**
<https://www.kaggle.com/c/nyc-taxi-trip-duration/data>

Download `train.csv` and place it at `data/train.csv`.

---

## Project structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py       — all constants, paths, hyper-parameters
│   └── features.py     — stateless feature engineering functions
├── train.py            — full training pipeline
├── predict.py          — inference entry point (Docker target)
├── artifacts/          — saved model + cluster centres (committed after training)
│   ├── lgbm_model.txt
│   ├── cluster_centers.npy
│   └── feature_cols.pkl
├── Dockerfile
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training

```bash
python train.py --data data/train.csv
```

Artefacts are written to `artifacts/`.  Training on the full 1.4 M row dataset
takes roughly 8–12 minutes on a 4-core CPU.

---

## Inference

```bash
python predict.py --input data/test.csv --output predictions.csv
```

Output is a CSV with two columns: `id` and `trip_duration` (integer seconds).

---

## Docker

**Build**
```bash
docker build -t eta-prediction .
```

**Run** (assumes artefacts are present in `artifacts/` before building)
```bash
docker run --rm \
  -v $(pwd)/data/test.csv:/data/test.csv \
  -v $(pwd)/output:/output \
  eta-prediction \
  --input /data/test.csv --output /output/predictions.csv
```

---

## Feature engineering

| Feature group | Features |
|---|---|
| Distance | `haversine_km`, `manhattan_km`, `abs_diff_lat`, `abs_diff_lon` |
| Spatial | `bearing`, `centre_lat/lon`, `pickup/dropoff_dist_centre` |
| Airport proximity | `pickup/dropoff_dist_jfk/lga/ewr` (6 features) |
| Datetime | `pickup_hour/minute/dow/month/doy/week` |
| Cyclic | `hour_sin/cos`, `dow_sin/cos` |
| Binary flags | `is_weekend`, `is_night`, `is_am_rush`, `is_pm_rush` |
| Location clusters | `pickup_cluster`, `dropoff_cluster`, `same_cluster` (K=100) |
| Raw | `vendor_id`, `passenger_count`, `store_and_fwd_flag` |
| Raw coordinates | `pickup/dropoff_latitude/longitude` |

---

## Model

LightGBM `LGBMRegressor` with `objective=regression_l1` (direct MAE
optimisation).  Validation split: 85/15, early stopping on 150 rounds with no
improvement.

Key hyper-parameters: `num_leaves=511`, `learning_rate=0.05`,
`feature_fraction=0.8`, `bagging_fraction=0.8`.

---

## What I tried

**Worked:**
- `regression_l1` objective rather than MSE — reduced val MAE by ~8 s compared
  to log-transform + MSE approach.
- K-Means location clusters at K=100 — added ~5 s MAE improvement over raw
  coordinates alone.
- Cyclic hour/DOW encoding — consistent 2–3 s improvement; prevents the model
  treating 23:00 and 00:00 as maximally different.

**Did not help:**
- Scaling the target with `np.log1p` — log-MSE optimisation underperforms
  direct MAE optimisation on the raw target for this metric.
- Adding `trip_distance` (from TLC-format data) — not present in the Kaggle
  2016 format; haversine distance is a sufficient proxy.
- CatBoost — comparable MAE to LightGBM but 3× slower to train.

**Next experiment:**
- Incorporate external traffic speed data (NYC DOT or Uber Movement) mapped to
  hour-of-day × borough cell — expected to reduce MAE by another 10–20 s on
  the 2024 evaluation slice where post-COVID commute patterns differ from 2016.
- Stacking a shallow MLP over LightGBM leaf embeddings.
