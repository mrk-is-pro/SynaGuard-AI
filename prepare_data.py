"""Phase 1a: inspect and clean raw sync logs. Works on any CSV with the 9 required columns."""
import json
import numpy as np
import pandas as pd
import config

NUM = ["master_price", "channel_price", "master_stock", "channel_stock", "units_sold", "sync_delay_min"]


def clean(df):
    """Returns (clean_df, report). Every fix is counted so it can be documented."""
    rep = {"rows_in": int(len(df))}
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["sku"] = d["sku"].astype(str).str.strip()
    d["channel"] = d["channel"].astype(str).str.strip()

    n = len(d)
    d = d.dropna(subset=["timestamp", "sku", "channel", "master_price", "master_stock"])
    rep["unusable_rows_dropped"] = int(n - len(d))
    n = len(d)
    d = d.drop_duplicates(subset=["sku", "channel", "timestamp"], keep="first")
    rep["duplicates_removed"] = int(n - len(d))

    bad = {"channel_stock": d["channel_stock"] < 0, "channel_price": d["channel_price"] <= 0,
           "sync_delay_min": d["sync_delay_min"] < 0, "units_sold": d["units_sold"] < 0}
    rep["invalid_values_fixed"] = {k: int(v.sum()) for k, v in bad.items()}
    for k, v in bad.items():
        d.loc[v, k] = np.nan                       # treat impossible values as missing
    rep["missing_values_filled"] = {c: int(d[c].isna().sum()) for c in NUM}

    ratio = (d["channel_price"] / d["master_price"]).groupby(d["channel"]).transform("median")
    d["channel_price"] = d["channel_price"].fillna((d["master_price"] * ratio).round(2))
    d["channel_stock"] = d["channel_stock"].fillna((d["master_stock"] / len(config.CHANNELS)).round())
    d["sync_delay_min"] = d["sync_delay_min"].fillna(d.groupby("channel")["sync_delay_min"].transform("median"))
    d["units_sold"] = d["units_sold"].fillna(0)

    d = d.sort_values(["sku", "channel", "timestamp"]).reset_index(drop=True)
    rep["rows_out"] = int(len(d))
    rep["skus"] = int(d["sku"].nunique())
    rep["channels"] = sorted(d["channel"].unique().tolist())
    rep["date_from"], rep["date_to"] = str(d["timestamp"].min()), str(d["timestamp"].max())
    return d, rep


def channel_stats(d):
    """Channel-level statistics of (mostly normal) behaviour."""
    g = d.assign(price_ratio=d["channel_price"] / d["master_price"],
                 stock_ratio=d["channel_stock"] / (d["master_stock"] / len(config.CHANNELS) + 1)).groupby("channel")
    out = pd.DataFrame({
        "Records": g.size(),
        "Avg price vs master": g["price_ratio"].median() - 1,
        "Price spread (std)": g["price_ratio"].std(),
        "Avg stock vs fair share": g["stock_ratio"].median(),
        "Avg sync delay (min)": g["sync_delay_min"].mean(),
        "95th pct delay (min)": g["sync_delay_min"].quantile(0.95),
        "Avg units sold/hr": g["units_sold"].mean(),
    })
    if "is_anomaly" in d.columns:
        out["Injected errors"] = g["is_anomaly"].sum()
    return out.round(3)


def encode(df):
    """Categorical + timestamp attributes encoded as numbers (WBS 1.3.2)."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["sku_code"] = d["sku"].astype("category").cat.codes
    d["channel_code"] = d["channel"].astype("category").cat.codes
    for ch in config.CHANNELS:
        d[f"channel_{ch.lower()}"] = (d["channel"] == ch).astype(int)
    status_map = {"success": 0, "retry": 1, "failed": 2}
    if "sync_status" in d:
        d["sync_status_code"] = d["sync_status"].map(status_map).fillna(0).astype(int)
    ts = d["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day = ts.dt.dayofweek
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    d["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    d["day_cos"] = np.cos(2 * np.pi * day / 7.0)
    d["epoch_sec"] = ts.astype("int64") // 10**9
    g = d.groupby(["sku", "channel"])
    gap = g["timestamp"].diff().dt.total_seconds() / 60.0
    d["update_gap_log"] = np.log1p(gap.fillna(60.0).clip(lower=0))
    return d


if __name__ == "__main__":
    raw = pd.read_csv(config.RAW_DATA_FILE)
    d, rep = clean(raw)
    d.to_csv(config.CLEAN_DATA_FILE, index=False)
    json.dump(rep, open(config.QUALITY_FILE, "w"), indent=2)
    print(f"Cleaned {rep['rows_in']} -> {rep['rows_out']} rows | duplicates {rep['duplicates_removed']} | "
          f"missing filled {sum(rep['missing_values_filled'].values())} | invalid fixed {sum(rep['invalid_values_fixed'].values())}")
    print(channel_stats(d).to_string())
