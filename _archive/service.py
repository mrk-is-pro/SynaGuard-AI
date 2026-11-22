"""Scoring logic shared by the API and the dashboard (one place, so both always agree)."""
import os
import pandas as pd
import config
from detector import Detector
from insights import enrich

_det = None


def ready():
    return all(os.path.exists(p) for p in (config.MODEL_FILE, config.SCALER_FILE, config.THRESHOLDS_FILE, config.BASELINE_FILE))


def get_detector():
    global _det
    if _det is None:
        _det = Detector()
    return _det


def score_records(records, sensitivity=1.0, use_rules=None, threshold=None):
    """records: list of dicts with the 9 sync fields. Returns one result dict per record, in input order.
    Note: the 'sudden change' features need history, so they are only computed between records of the
    same SKU and channel inside one request (a single record has no previous record)."""
    df = pd.DataFrame(records).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    det = get_detector()
    res = enrich(det.apply_settings(det.score(df), sensitivity=sensitivity, use_rules=use_rules,
                                    threshold=threshold)).sort_index()
    thr = (threshold or det.threshold) / sensitivity
    return [{
        "index": int(i), "sku": r["sku"], "channel": r["channel"], "timestamp": r["timestamp"].isoformat(),
        "anomaly_score": round(float(r["score"]), 4), "threshold": round(float(thr), 4),
        "is_anomaly": bool(r["flagged"]), "severity": r["severity"],
        "likely_type": r["likely_type"] if r["flagged"] else None,
        "reason": r["reason"] if r["flagged"] else None,
        "detected_by": r["source"] if r["flagged"] else None,
        "estimated_revenue_at_risk": float(r["est_impact"]) if r["flagged"] else 0.0,
        "suggested_action": r["action"] if r["flagged"] else None,
        "feature_scores": {c: round(float(r["err_" + c]), 3) for c in config.FEATURE_COLUMNS},
    } for i, r in res.iterrows()]
