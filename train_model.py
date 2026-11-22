"""Phase 2: train, pick threshold(s), evaluate on held-out data against the hidden labels."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config
import autoencoder as ae
import thresholds as th

df = pd.read_csv(config.FEATURES_FILE)
train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED)
F = config.FEATURE_COLUMNS

# Labels are NOT used for training, only for the final report card.
X_train = train_df[F]
model, keep = ae.train_clean(X_train)
ae.save(model)

clean_scores = ae.anomaly_scores(model, X_train[keep])          # mostly-normal training rows
methods = th.compute_methods(clean_scores)
train_scores = ae.anomaly_scores(model, X_train)
methods["f1_optimal"] = th.f1_optimal(train_scores, train_df["is_anomaly"].values)
chosen = methods[config.THRESHOLD_METHOD]

test_scores = ae.anomaly_scores(model, test_df[F])
y = test_df["is_anomaly"].values
evaluation = [{"method": k, "name": th.NAMES[k], **th.evaluate(test_scores, y, v)} for k, v in methods.items()]
json.dump({"chosen_method": config.THRESHOLD_METHOD, "chosen": chosen, "methods": methods,
           "evaluation": evaluation}, open(config.THRESHOLDS_FILE, "w"), indent=2)

pred = (test_scores > chosen).astype(int)
m = th.evaluate(test_scores, y, chosen)
tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
res = test_df.assign(pred=pred)
per_type = res[res.anomaly_type != "none"].groupby("anomaly_type")["pred"].mean().round(3).to_dict()
json.dump({**m, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall_by_type": per_type,
           "n_train": int(len(train_df)), "n_test": int(len(test_df)), "trimmed_rows": int((~keep).sum()),
           "iterations": int(model.n_iter_), "final_loss": float(model.loss_)},
          open(config.METRICS_FILE, "w"), indent=2)

print(f"Chosen threshold ({config.THRESHOLD_METHOD}): {chosen:.4f}")
print(f"Precision {m['precision']:.3f}  Recall {m['recall']:.3f}  F1 {m['f1']:.3f}  | caught {tp}, missed {fn}, false alarms {fp}")
print("Recall by type:", per_type)
print(pd.DataFrame(evaluation)[["name", "threshold", "alerts", "precision", "recall", "f1"]].round(3).to_string(index=False))
