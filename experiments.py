"""Phase 2b: compare autoencoder architectures (AI/ML experimentation). Uses a 20k-row sample for speed."""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config
import autoencoder as ae
import thresholds as th

ARCHS = {"6 → 3 → 6": (3,), "6 → 4 → 6": (4,), "6 → 6 → 6 (no bottleneck)": (6,),
         "6 → 8 → 4 → 8 → 6": (8, 4, 8), "6 → 8 → 4 → 2 → 4 → 8 → 6": (8, 4, 2, 4, 8)}

df = pd.read_csv(config.FEATURES_FILE)
train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED)
train_df = train_df.sample(min(20000, len(train_df)), random_state=config.SEED)
F, rows = config.FEATURE_COLUMNS, []
normal = test_df[test_df["is_anomaly"] == 0][F]
for name, hidden in ARCHS.items():
    t0 = time.time()
    model, keep = ae.train_clean(train_df[F], hidden)
    thr = th.compute_methods(ae.anomaly_scores(model, train_df[F][keep]))["percentile"]
    r = th.evaluate(ae.anomaly_scores(model, test_df[F]), test_df["is_anomaly"].values, thr)
    rows.append({"architecture": name, "hidden": str(hidden), "chosen": hidden == config.HIDDEN_LAYERS,
                 "normal_mse": float(((normal.values - model.predict(normal)) ** 2).mean()),
                 "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                 "seconds": round(time.time() - t0, 1), "iterations": int(model.n_iter_)})
out = pd.DataFrame(rows)
out.to_csv(config.EXPERIMENTS_FILE, index=False)
print(out.round(3).to_string(index=False))
