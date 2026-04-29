"""
predict.py — NYC Taxi ETA: inference entry point.

Usage
-----
    python predict.py --input data/test.csv --output predictions.csv

The script loads the trained LightGBM model and cluster artefacts from
artifacts/, runs feature engineering on the input CSV, and writes a
two-column CSV (id, trip_duration) to the specified output path.

Column requirements for --input:
    id, vendor_id, pickup_datetime, passenger_count,
    pickup_longitude, pickup_latitude,
    dropoff_longitude, dropoff_latitude,
    store_and_fwd_flag
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from src.config import ARTIFACTS_DIR, CLUSTER_PATH, MODEL_PATH
from src.features import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Artefact loading ───────────────────────────────────────────────────────────

def load_model() -> lgb.Booster:
    if not MODEL_PATH.exists():
        log.error("Model artefact not found: %s", MODEL_PATH)
        sys.exit(1)
    return lgb.Booster(model_file=str(MODEL_PATH))


def load_cluster_model() -> MiniBatchKMeans | None:
    if not CLUSTER_PATH.exists():
        log.warning("Cluster artefact not found — cluster features will be skipped.")
        return None

    centres = np.load(CLUSTER_PATH)
    km = MiniBatchKMeans(n_clusters=len(centres), n_init=1)
    # Inject precomputed centres so .predict() works without re-fitting
    km.cluster_centers_ = centres
    km._n_threads = 1
    return km


def load_feature_cols() -> list[str] | None:
    meta_path = ARTIFACTS_DIR / "feature_cols.pkl"
    if not meta_path.exists():
        return None
    with open(meta_path, "rb") as fh:
        return pickle.load(fh)


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(input_path: Path, output_path: Path) -> None:
    log.info("Loading input: %s", input_path)
    df = pd.read_csv(input_path)
    log.info("Rows to predict: %d", len(df))

    model = load_model()
    cluster_model = load_cluster_model()
    feature_cols = load_feature_cols()

    df_feat = build_features(df, cluster_model=cluster_model)

    if feature_cols is not None:
        cols = [c for c in feature_cols if c in df_feat.columns]
        missing = set(feature_cols) - set(df_feat.columns)
        if missing:
            log.warning("Missing features (filling with 0): %s", missing)
            for col in missing:
                df_feat[col] = 0
        X = df_feat[feature_cols]
    else:
        # Fallback: use whatever numeric columns are available
        X = df_feat.select_dtypes(include=[np.number]).drop(
            columns=["trip_duration"], errors="ignore"
        )

    preds = model.predict(X)
    preds = np.clip(preds, 1, None)  # duration cannot be negative

    result = pd.DataFrame({
        "id": df["id"] if "id" in df.columns else range(len(df)),
        "trip_duration": preds.astype(int),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info("Predictions written → %s  (rows: %d)", output_path, len(result))
    log.info("Prediction stats — mean: %.1fs  median: %.1fs  p95: %.1fs",
             preds.mean(), np.median(preds), np.percentile(preds, 95))


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC taxi ETA inference")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Path to write output CSV  (default: predictions.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    predict(args.input, args.output)


if __name__ == "__main__":
    main()
