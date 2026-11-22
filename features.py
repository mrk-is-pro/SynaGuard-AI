"""Phase 1b: turn clean logs into numeric features the autoencoder can learn from."""
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import config

FEATURE_INFO = pd.DataFrame([
    ("price_diff_pct", "(channel price / master price) / channel's normal ratio − 1", "Catches wrong or stale prices and decimal slips"),
    ("stock_diff_pct", "(channel stock − fair share of master stock) / fair share", "Catches inventory that drifted from the master"),
    ("sync_delay_log", "log(1 + minutes since last sync)", "Catches late or stopped integrations (log tames the long tail)"),
    ("stock_cover_hours", "log(1 + channel stock / units sold per hour)", "Catches oversell risk: stock running out faster than expected"),
    ("price_change_pct", "channel price % change minus master price % change since the previous record (capped)", "Catches sudden price jumps the master did not make"),
    ("stock_change_pct", "channel stock % change minus master stock % change since the previous record (capped)", "Catches sudden stock jumps the master did not make (shared restocks cancel out)"),
    ("sync_attempts_log", "log(1 + number of attempts the sync needed)", "Catches integrations that keep retrying"),
    ("update_gap_log", "log(1 + minutes since the previous update of the same product and channel)", "Catches irregular update intervals and missing updates"),
    ("sync_status_code", "0 = success, 1 = retry, 2 = failed (categorical, encoded as a number)", "Catches syncs the integration itself reported as failing"),
], columns=["Feature", "How it is computed", "Why it matters"])


def compute_baseline(df):
    """What 'normal' looks like per channel (e.g. Amazon prices run ~3% above master)."""
    return {"price_ratio": (df["channel_price"] / df["master_price"]).groupby(df["channel"]).median().to_dict()}


def build_features(df, baseline=None):
    """Raw rows -> feature table (unscaled). Used for training, the dashboard and the API."""
    df = df.sort_values(["sku", "channel", "timestamp"]).copy()
    g = df.groupby(["sku", "channel"])
    ratio = df["channel_price"] / df["master_price"]
    if baseline:
        expected_ratio = df["channel"].map(baseline["price_ratio"]).fillna(ratio.median())
    else:
        expected_ratio = ratio.groupby(df["channel"]).transform("median")
    out = pd.DataFrame(index=df.index)
    out["price_diff_pct"] = ratio / expected_ratio - 1
    expected_stock = df["master_stock"] / len(config.CHANNELS)
    out["stock_diff_pct"] = (df["channel_stock"] - expected_stock) / (expected_stock + 1)
    out["sync_delay_log"] = np.log1p(df["sync_delay_min"].clip(lower=0))
    sales_rate = df["units_sold"].clip(lower=1)
    out["stock_cover_hours"] = np.log1p(df["channel_stock"].clip(lower=0) / sales_rate)
    ch_p_pct = g["channel_price"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    m_p_pct = g["master_price"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    out["price_change_pct"] = (ch_p_pct - m_p_pct).clip(-1, 3)
    ch_s_pct = g["channel_stock"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    m_s_pct = g["master_stock"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    out["stock_change_pct"] = (ch_s_pct - m_s_pct).clip(-1, 3)
    attempts = df["sync_attempts"] if "sync_attempts" in df else 1
    out["sync_attempts_log"] = np.log1p(attempts.clip(lower=0))
    gap = g["timestamp"].diff().dt.total_seconds() / 60.0
    out["update_gap_log"] = np.log1p(gap.fillna(60.0).clip(lower=0))
    status_map = {"success": 0, "retry": 1, "failed": 2}
    status = df["sync_status"] if "sync_status" in df else "success"
    out["sync_status_code"] = status.map(status_map).fillna(0).astype(float)
    return out[config.FEATURE_COLUMNS], df


def scale_and_save():
    raw = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=["timestamp"])
    baseline = compute_baseline(raw)
    feats, raw = build_features(raw, baseline)
    scaler = StandardScaler().fit(feats)
    scaled = pd.DataFrame(scaler.transform(feats), columns=feats.columns, index=feats.index)
    scaled["is_anomaly"] = raw["is_anomaly"].values
    scaled["anomaly_type"] = raw["anomaly_type"].values
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, config.SCALER_FILE)
    json.dump(baseline, open(config.BASELINE_FILE, "w"), indent=2)
    scaled.to_csv(config.FEATURES_FILE, index=False)
    print(f"Saved {len(scaled)} rows -> {config.FEATURES_FILE}")
    print(feats.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    scale_and_save()
