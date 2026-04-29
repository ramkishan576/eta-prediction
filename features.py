"""
Feature engineering for NYC taxi trip duration prediction.

All transformations are pure functions that operate on a DataFrame and return
a new DataFrame.  No side effects; no global state.
"""

import numpy as np
import pandas as pd

from src.config import AIRPORTS


# ── Haversine distance ─────────────────────────────────────────────────────────

def haversine(lat1: np.ndarray, lon1: np.ndarray,
              lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Return great-circle distance in kilometres."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Bearing ────────────────────────────────────────────────────────────────────

def bearing(lat1: np.ndarray, lon1: np.ndarray,
            lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compass bearing from origin to destination (degrees, 0–360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    x = np.sin(dlambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ── Airport proximity ──────────────────────────────────────────────────────────

def airport_distance_km(lat: np.ndarray, lon: np.ndarray,
                        airport_lat: float, airport_lon: float) -> np.ndarray:
    return haversine(lat, lon,
                     np.full_like(lat, airport_lat, dtype=float),
                     np.full_like(lon, airport_lon, dtype=float))


# ── Core feature builder ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, cluster_model=None) -> pd.DataFrame:
    """
    Transform raw input DataFrame into a model-ready feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns expected by the challenge harness.
        Accepts both Kaggle-2016 column names and TLC column names.
    cluster_model : fitted sklearn KMeans | None
        When supplied, appends cluster-ID features for pickup and dropoff.

    Returns
    -------
    pd.DataFrame
        Feature matrix (no target column).
    """
    df = df.copy()

    # ── Normalise column names ─────────────────────────────────────────────────
    df = _normalise_columns(df)

    # ── Datetime decomposition ─────────────────────────────────────────────────
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"]       = dt.dt.hour
    df["pickup_minute"]     = dt.dt.minute
    df["pickup_dow"]        = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["pickup_month"]      = dt.dt.month
    df["pickup_doy"]        = dt.dt.dayofyear
    df["pickup_week"]       = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]        = (df["pickup_dow"] >= 5).astype(np.int8)
    df["is_night"]          = ((df["pickup_hour"] >= 22) |
                               (df["pickup_hour"] < 6)).astype(np.int8)
    df["is_am_rush"]        = ((df["pickup_hour"] >= 7) &
                               (df["pickup_hour"] <= 9)).astype(np.int8)
    df["is_pm_rush"]        = ((df["pickup_hour"] >= 16) &
                               (df["pickup_hour"] <= 19)).astype(np.int8)

    # Hour expressed as a cyclic pair so midnight ≈ 23:00
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["pickup_dow"]  /  7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["pickup_dow"]  /  7)

    # ── Spatial features ───────────────────────────────────────────────────────
    plat = df["pickup_latitude"].values
    plon = df["pickup_longitude"].values
    dlat = df["dropoff_latitude"].values
    dlon = df["dropoff_longitude"].values

    df["haversine_km"]   = haversine(plat, plon, dlat, dlon)
    df["abs_diff_lat"]   = np.abs(dlat - plat)
    df["abs_diff_lon"]   = np.abs(dlon - plon)
    df["manhattan_km"]   = (df["abs_diff_lat"] * 111.0 +
                            df["abs_diff_lon"] * 85.0)   # rough km/degree
    df["bearing"]        = bearing(plat, plon, dlat, dlon)
    df["centre_lat"]     = (plat + dlat) / 2
    df["centre_lon"]     = (plon + dlon) / 2

    # Distance from NYC centre (Columbus Circle)
    nyc_lat, nyc_lon = 40.7681, -73.9819
    df["pickup_dist_centre"]  = haversine(plat, plon,
                                          np.full_like(plat, nyc_lat),
                                          np.full_like(plon, nyc_lon))
    df["dropoff_dist_centre"] = haversine(dlat, dlon,
                                          np.full_like(dlat, nyc_lat),
                                          np.full_like(dlon, nyc_lon))

    # Airport proximity
    for name, (a_lat, a_lon) in AIRPORTS.items():
        df[f"pickup_dist_{name}"]  = airport_distance_km(plat, plon, a_lat, a_lon)
        df[f"dropoff_dist_{name}"] = airport_distance_km(dlat, dlon, a_lat, a_lon)

    # ── Cluster features ───────────────────────────────────────────────────────
    if cluster_model is not None:
        pickup_coords  = np.column_stack([plat, plon])
        dropoff_coords = np.column_stack([dlat, dlon])
        df["pickup_cluster"]  = cluster_model.predict(pickup_coords).astype(np.int16)
        df["dropoff_cluster"] = cluster_model.predict(dropoff_coords).astype(np.int16)
        df["same_cluster"]    = (
            df["pickup_cluster"] == df["dropoff_cluster"]
        ).astype(np.int8)

    # ── Passenger count ────────────────────────────────────────────────────────
    df["passenger_count"] = df["passenger_count"].fillna(1).clip(1, 6)

    # ── Store-and-forward flag ─────────────────────────────────────────────────
    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = (
            df["store_and_fwd_flag"].map({"Y": 1, "N": 0}).fillna(0).astype(np.int8)
        )

    # ── Vendor ID ──────────────────────────────────────────────────────────────
    if "vendor_id" in df.columns:
        df["vendor_id"] = pd.to_numeric(df["vendor_id"], errors="coerce").fillna(1).astype(np.int8)

    return df


# ── Column name normalisation ─────────────────────────────────────────────────

_COLUMN_MAP = {
    # TLC 2024 → canonical
    "VendorID":             "vendor_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "PULocationID":         "pu_location_id",
    "DOLocationID":         "do_location_id",
}

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in _COLUMN_MAP.items() if k in df.columns})


# ── Feature column list (used by train.py and predict.py) ─────────────────────

BASE_FEATURE_COLS = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_minute",
    "pickup_dow",
    "pickup_month",
    "pickup_doy",
    "pickup_week",
    "is_weekend",
    "is_night",
    "is_am_rush",
    "is_pm_rush",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "haversine_km",
    "abs_diff_lat",
    "abs_diff_lon",
    "manhattan_km",
    "bearing",
    "centre_lat",
    "centre_lon",
    "pickup_dist_centre",
    "dropoff_dist_centre",
    "pickup_dist_jfk",
    "dropoff_dist_jfk",
    "pickup_dist_lga",
    "dropoff_dist_lga",
    "pickup_dist_ewr",
    "dropoff_dist_ewr",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]

CLUSTER_FEATURE_COLS = [
    "pickup_cluster",
    "dropoff_cluster",
    "same_cluster",
]
