"""
train.py — NYC Taxi ETA: end-to-end training pipeline.

Usage
-----
    python train.py --data data/train.csv

The script:
  1. Loads and cleans the Kaggle NYC Taxi Trip Duration CSV.
  2. Fits a K-Means location-cluster model and saves its centres.
  3. Builds the full feature matrix.
  4. Trains a LightGBM regressor (MAE objective) with early stopping.
  5. Reports hold-out MAE and saves the model to artifacts/.
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    CLUSTER_PATH,
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    MODEL_PATH,
    N_LOCATION_CLUSTERS,
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LON_MAX,
    NYC_LON_MIN,
    PASSENGER_COUNT_MAX,
    PASSENGER_COUNT_MIN,
    RANDOM_STATE,
    TRIP_DURATION_MAX,
    TRIP_DURATION_MIN,
)
from src.features import BASE_FEATURE_COLS, CLUSTER_FEATURE_COLS, build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    log.info("Loading %s", path)
    df = pd.read_csv(path, parse_dates=["pickup_datetime"])
    log.info("Rows loaded: %d", len(df))
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Remove rows with missing coordinates or target
    df = df.dropna(subset=[
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "trip_duration",
    ])

    # Restrict to NYC bounding box
    df = df[
        df["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX) &
        df["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX) &
        df["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX) &
        df["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)
    ]

    # Target range filter
    df = df[df["trip_duration"].between(TRIP_DURATION_MIN, TRIP_DURATION_MAX)]

    # Passenger count filter
    df = df[df["passenger_count"].between(PASSENGER_COUNT_MIN, PASSENGER_COUNT_MAX)]

    after = len(df)
    log.info("Cleaning removed %d rows (%d remaining)", before - after, after)
    return df.reset_index(drop=True)


# ── Location cluster model ─────────────────────────────────────────────────────

def fit_location_clusters(df: pd.DataFrame) -> MiniBatchKMeans:
    log.info("Fitting %d-cluster location model", N_LOCATION_CLUSTERS)
    coords = np.vstack([
        df[["pickup_latitude", "pickup_longitude"]].values,
        df[["dropoff_latitude", "dropoff_longitude"]].values,
    ])
    kmeans = MiniBatchKMeans(
        n_clusters=N_LOCATION_CLUSTERS,
        batch_size=10_000,
        random_state=RANDOM_STATE,
        n_init=3,
    )
    kmeans.fit(coords)
    np.save(CLUSTER_PATH, kmeans.cluster_centers_)
    log.info("Cluster centres saved → %s", CLUSTER_PATH)
    return kmeans


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, cluster_model: MiniBatchKMeans):
    df = build_features(df, cluster_model=cluster_model)

    feature_cols = BASE_FEATURE_COLS + CLUSTER_FEATURE_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["trip_duration"].values
    return X, y, feature_cols


# ── Training ───────────────────────────────────────────────────────────────────

def train(X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMRegressor:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    log.info("Train size: %d | Validation size: %d", len(X_tr), len(X_val))

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.time() - t0

    val_preds = model.predict(X_val)
    mae = np.mean(np.abs(val_preds - y_val))
    log.info(
        "Training complete in %.1fs | Best iteration: %d | Val MAE: %.2fs",
        elapsed, model.best_iteration_, mae,
    )
    return model


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model: lgb.LGBMRegressor, feature_cols: list[str]) -> None:
    model.booster_.save_model(str(MODEL_PATH))

    meta_path = ARTIFACTS_DIR / "feature_cols.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump(feature_cols, fh)

    log.info("Model saved → %s", MODEL_PATH)
    log.info("Feature list saved → %s", meta_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NYC taxi ETA model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to raw training CSV (Kaggle NYC Taxi Trip Duration format)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        log.error("Data file not found: %s", args.data)
        sys.exit(1)

    df = load_raw(args.data)
    df = clean(df)

    cluster_model = fit_location_clusters(df)
    X, y, feature_cols = prepare_features(df, cluster_model)

    model = train(X, y)
    save_model(model, feature_cols)

    log.info("Pipeline finished — artifacts in %s/", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
