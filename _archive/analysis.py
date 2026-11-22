"""Shared analysis used by both the dashboard lab and the HTML reports."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import config


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def artifacts():
    exp = pd.read_csv(config.EXPERIMENTS_FILE) if os.path.exists(config.EXPERIMENTS_FILE) else None
    hist = pd.read_csv(config.HISTORY_FILE) if os.path.exists(config.HISTORY_FILE) else None
    return {"history": hist, "quality": load_json(config.QUALITY_FILE), "thresholds": load_json(config.THRESHOLDS_FILE),
            "metrics": load_json(config.METRICS_FILE), "experiments": exp}


def _prf(y, p):
    return (precision_score(y, p, zero_division=0), recall_score(y, p, zero_division=0), f1_score(y, p, zero_division=0))


def add_context(res):
    """Marks rows that directly follow a real error (needs labels; used only for analysis)."""
    if "is_anomaly" in res.columns:
        res = res.copy()
        res["after_error"] = res.groupby(["sku", "channel"])["is_anomaly"].shift(1).fillna(0).astype(bool)
    return res


def recovery_effect(det, scored, settings):
    """Held-out accuracy with and without recovery suppression."""
    from detector import holdout
    rows = []
    for label, flag in [("Off (alert on every unusual sync)", False), ("On (skip the sync right after an alert)", True)]:
        h = holdout(add_context(det.apply_settings(scored, **{**settings, "suppress_recovery": flag})))
        p, r, f = _prf(h["is_anomaly"], h["flagged"].astype(int))
        rows.append({"Recovery suppression": label, "Alerts": int(h["flagged"].sum()),
                     "False alarms": int((h["flagged"] & (h["is_anomaly"] == 0)).sum()),
                     "Precision": p, "Recall": r, "F1": f})
    return pd.DataFrame(rows)


def comparison_table(h):
    y = h["is_anomaly"]
    rows = []
    for name, col in [("Rules only (traditional monitoring)", "rule_flag"), ("Autoencoder only", "ae_flag"),
                      ("Autoencoder + rules (hybrid)", "flagged")]:
        p = h[col].astype(int)
        pr, rc, f = _prf(y, p)
        rows.append({"Approach": name, "Alerts": int(p.sum()), "Precision": pr, "Recall": rc, "F1": f})
    return pd.DataFrame(rows)


def type_comparison(h):
    e = h[h["is_anomaly"] == 1]
    t = pd.DataFrame({"Rules only": e.groupby("anomaly_type")["rule_flag"].mean(),
                      "Autoencoder only": e.groupby("anomaly_type")["ae_flag"].mean(),
                      "Hybrid": e.groupby("anomaly_type")["flagged"].mean()})
    return t.fillna(0)


def fp_analysis(h, thr):
    """False positives / false negatives: what triggered them and how they differ from true alerts."""
    normal = h["is_anomaly"] == 0
    fp = h[h["flagged"] & normal]
    fn = h[~h["flagged"] & (h["is_anomaly"] == 1)]
    tp = h[h["flagged"] & ~normal]
    tn = h[~h["flagged"] & normal]
    err = ["err_" + c for c in config.FEATURE_COLUMNS]
    groups = pd.DataFrame({"Normal, left alone": tn[err].mean(), "False alarm": fp[err].mean(),
                           "Real error, caught": tp[err].mean(), "Real error, missed": fn[err].mean()}).T.clip(upper=50)
    groups.columns = config.FEATURE_COLUMNS
    reasons = fp["reason"].value_counts()
    border = float((fp["score"] < 2 * thr).mean()) if len(fp) else 0.0
    missed_types = fn["anomaly_type"].value_counts()
    after = float(fp["after_error"].mean()) if len(fp) and "after_error" in fp else 0.0
    lines = [f"{len(fp):,} false alarms, which is {len(fp) / max(normal.sum(), 1):.2%} of normal syncs."]
    if len(fp):
        lines.append(f"The most common trigger was “{reasons.index[0]}” ({reasons.iloc[0] / len(fp):.0%} of false alarms).")
        lines.append(f"{border:.0%} of false alarms scored below twice the threshold, so they were borderline cases.")
    if after >= 0.2:
        lines.append(f"{after:.0%} of false alarms happened in the record right after a real error: when a price or stock count snaps "
                     "back to normal it can look unusual; the recovery suppression setting targets exactly these cases.")
    if len(fn):
        lines.append(f"{len(fn):,} real errors were missed; most were “{missed_types.index[0]}” ({missed_types.iloc[0]}).")
    return {"fp": fp, "fn": fn, "tp": tp, "groups": groups, "reasons": reasons, "by_channel": fp["channel"].value_counts(),
            "by_source": fp["source"].value_counts(), "summary": lines, "borderline": border, "after_error": after}


CAUSES = {"promo": "Legitimate promotion (channel catching up)", "bulk_update": "Bulk inventory update"}


def fp_causes(h):
    """What explains the false alarms? Known legitimate events versus no known cause."""
    fp = h[h["flagged"] & (h["is_anomaly"] == 0)]
    ev = fp["event"].map(CAUSES) if "event" in fp else pd.Series(dtype=object)
    return ev.fillna("No known cause").value_counts()


def cross_channel(res):
    """Do price/stock differences BETWEEN channels show up in the anomaly scores? (WBS 4.2.2)"""
    d = res.copy()
    d["hr"] = d["timestamp"].dt.floor("h")
    d["ratio"] = d["channel_price"] / d["master_price"]
    g = d.groupby(["sku", "hr"])["ratio"]
    d["price_spread"] = g.transform("max") - g.transform("min")
    d["sr"] = d["channel_stock"] / (d["master_stock"] / len(config.CHANNELS) + 1)
    g = d.groupby(["sku", "hr"])["sr"]
    d["stock_spread"] = g.transform("max") - g.transform("min")
    d["spread"] = np.maximum(d["price_spread"] / 0.10, d["stock_spread"] / 0.5)
    labels = ["Aligned", "Slight", "Noticeable", "Large", "Very different"]
    d["band"] = pd.cut(d["spread"], [-0.001, 0.25, 0.5, 1, 2, np.inf], labels=labels)
    return d.groupby("band", observed=True).agg(rows=("score", "size"), alerts=("flagged", "mean"),
                                                 real_errors=("is_anomaly", "mean") if "is_anomaly" in d else ("flagged", "mean"),
                                                 median_score=("score", "median"))


def suspicious(res, n=10):
    """Highest-scoring events with their likely cause (WBS 3.1.2 / 3.1.3)."""
    from insights import enrich
    top = enrich(res[res["flagged"]]).sort_values("score", ascending=False)
    cols = ["timestamp", "sku", "channel", "likely_type", "reason", "score", "est_impact"]
    return top.head(n)[cols], top["likely_type"].value_counts()
