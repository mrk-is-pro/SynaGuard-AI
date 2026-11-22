"""Phase 1a: create fake-but-realistic sync logs with hidden, labeled errors."""
import os
import numpy as np
import pandas as pd
import config


def generate():
    rng = np.random.default_rng(config.SEED)
    base_price = rng.uniform(10, 200, config.N_SKUS).round(2)
    base_stock = rng.integers(200, 1000, config.N_SKUS)
    n_steps = config.DAYS * config.RECORDS_PER_DAY
    times = pd.date_range("2026-01-01", periods=n_steps, freq="h")
    rows = []

    for s in range(config.N_SKUS):
        sku = f"SKU{s:03d}"
        master_stock = float(base_stock[s])
        for ch, markup in config.CHANNELS.items():
            share = 1 / len(config.CHANNELS)
            for t in times:
                master_price = base_price[s] * (1 + 0.0002 * rng.normal())
                sold = rng.poisson(max(master_stock * 0.01, 1))
                master_stock = max(master_stock - sold, 0)
                if master_stock < 50:            # restock
                    master_stock += rng.integers(300, 800)
                rows.append({
                    "timestamp": t, "sku": sku, "channel": ch,
                    "master_price": round(master_price, 2),
                    "channel_price": round(master_price * (1 + markup + 0.002 * rng.normal()), 2),
                    "master_stock": master_stock,
                    "channel_stock": round(master_stock * share + rng.normal(0, 1)),
                    "units_sold": sold,
                    "sync_delay_min": rng.exponential(5),
                    "is_anomaly": 0, "anomaly_type": "none",
                })

    df = pd.DataFrame(rows)
    df = inject_anomalies(df, rng)
    return add_messiness(df, rng)


def inject_anomalies(df, rng):
    """Corrupt a random 2% of rows; record what we did (ground truth)."""
    n = int(len(df) * config.ANOMALY_RATE)
    idx = rng.choice(df.index, n, replace=False)
    types = ["price_mismatch", "decimal_error", "stale_sync", "stock_drift", "oversell_risk"]
    for i in idx:
        kind = rng.choice(types)
        if kind == "price_mismatch":
            df.at[i, "channel_price"] = round(df.at[i, "channel_price"] * rng.choice([0.6, 0.75, 1.3, 1.5]), 2)
        elif kind == "decimal_error":
            df.at[i, "channel_price"] = round(df.at[i, "channel_price"] / 10, 2)
        elif kind == "stale_sync":
            df.at[i, "sync_delay_min"] = rng.uniform(120, 600)
        elif kind == "stock_drift":
            df.at[i, "channel_stock"] = round(df.at[i, "channel_stock"] * rng.choice([0.3, 0.5, 2.0]))
        elif kind == "oversell_risk":
            df.at[i, "channel_stock"] = rng.integers(0, 3)
            df.at[i, "units_sold"] = df.at[i, "units_sold"] + rng.integers(10, 30)
        df.at[i, "is_anomaly"] = 1
        df.at[i, "anomaly_type"] = kind
    return df


def add_messiness(df, rng):
    """Make the export look like real operational data: duplicates, gaps, impossible values, shuffled order."""
    n = max(int(len(df) * config.MESSY_RATE), 4)
    df = pd.concat([df, df.sample(n, random_state=config.SEED)], ignore_index=True)       # duplicate rows
    idx = rng.choice(df.index, 3 * n, replace=False)
    df.loc[idx[:n], "channel_price"] = np.nan                                              # missing prices
    df.loc[idx[n:2 * n], "sync_delay_min"] = np.nan                                        # missing delays
    df.loc[idx[2 * n:2 * n + n // 2], "channel_stock"] = -rng.integers(1, 20, n // 2)      # negative stock
    df.loc[idx[2 * n + n // 2:], "channel_stock"] = np.nan                                 # missing stock
    return df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)              # unsorted export


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = generate()
    data.to_csv(config.RAW_DATA_FILE, index=False)
    print(f"Saved {len(data)} raw rows -> {config.RAW_DATA_FILE}")
