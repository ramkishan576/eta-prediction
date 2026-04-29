"""
Central configuration for the NYC taxi ETA pipeline.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "lgbm_model.txt"
CLUSTER_PATH = ARTIFACTS_DIR / "cluster_centers.npy"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

# ── Data constraints ───────────────────────────────────────────────────────────
# NYC bounding box (lon_min, lon_max, lat_min, lat_max)
NYC_LON_MIN = -74.05
NYC_LON_MAX = -73.75
NYC_LAT_MIN = 40.63
NYC_LAT_MAX = 40.85

TRIP_DURATION_MIN = 60        # seconds — remove sub-minute trips
TRIP_DURATION_MAX = 7_200     # seconds — remove trips > 2 hours
PASSENGER_COUNT_MIN = 1
PASSENGER_COUNT_MAX = 6

# ── Feature engineering ────────────────────────────────────────────────────────
N_LOCATION_CLUSTERS = 100     # K-Means clusters on pickup/dropoff coords
RANDOM_STATE = 42

# ── LightGBM hyper-parameters ─────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "regression_l1",   # optimise MAE directly
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 511,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 3000,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}

EARLY_STOPPING_ROUNDS = 150

# ── Notable NYC airport coordinates ───────────────────────────────────────────
AIRPORTS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
}
