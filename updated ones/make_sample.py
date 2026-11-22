"""Compact the cleaned project dataset into the sample CSV embedded in the dashboard."""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config  # noqa: E402

d = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=["timestamp"])
d["timestamp"] = d["timestamp"].dt.strftime("%Y-%m-%dT%H:%M")
for c in ["master_price", "channel_price"]:
    d[c] = d[c].round(2)
d["sync_delay_min"] = d["sync_delay_min"].round(1)
cols = ["timestamp", "sku", "channel", "master_price", "channel_price", "master_stock", "channel_stock",
        "units_sold", "sync_delay_min", "sync_attempts", "sync_status", "is_anomaly"]
out = os.path.join(os.path.dirname(__file__), "generated", "sample.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
d[cols].astype({"master_stock": int, "channel_stock": int, "units_sold": int, "sync_attempts": int}).to_csv(out, index=False)
print("sample.csv", round(os.path.getsize(out) / 1e6, 2), "MB")
