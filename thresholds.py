"""Phase 3a: different ways to choose the alert threshold, and how each performs."""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import config

NAMES = {"percentile": f"Percentile ({config.THRESHOLD_PERCENTILE}th of normal scores)",
         "mean_3sd": "Mean + 3 standard deviations",
         "mad": "Median + 3 × MAD (robust)",
         "iqr": "Q3 + 3 × IQR (box-plot rule)",
         "f1_optimal": "Best F1 on training labels (reference only)"}


def compute_methods(clean_scores):
    s = np.asarray(clean_scores)
    q1, q3 = np.percentile(s, [25, 75])
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826
    return {"percentile": float(np.percentile(s, config.THRESHOLD_PERCENTILE)),
            "mean_3sd": float(s.mean() + 3 * s.std()),
            "mad": float(med + 3 * mad),
            "iqr": float(q3 + 3 * (q3 - q1))}


def f1_optimal(scores, y):
    """Uses labels, so it is only a yardstick, never the live method."""
    cands = np.unique(np.percentile(scores, np.linspace(90, 99.95, 150)))
    best = max(cands, key=lambda t: f1_score(y, (scores > t).astype(int), zero_division=0))
    return float(best)


def evaluate(scores, y, thr):
    p = (np.asarray(scores) > thr).astype(int)
    return {"threshold": float(thr), "alerts": int(p.sum()),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0))}
