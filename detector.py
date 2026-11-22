"""Phase 3: the detection engine. Give it raw sync rows, get back alerts with reasons.
score() is the slow part (runs the model). apply_settings() is instant, so the dashboard
sliders can change sensitivity and rules live."""
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import config
import autoencoder as ae
from features import build_features

warnings.filterwarnings("ignore", message="obj.round")


class Detector:
    def __init__(self):
        self.model = ae.load()
        self.scaler = joblib.load(config.SCALER_FILE)
        t = json.load(open(config.THRESHOLDS_FILE))
        self.threshold, self.methods = t["chosen"], t["methods"]
        self.baseline = json.load(open(config.BASELINE_FILE))

    def score(self, raw):
        """Run the model. Adds score, price_diff_pct, per-feature errors (err_*)."""
        feats, raw = build_features(raw, self.baseline)
        X = pd.DataFrame(self.scaler.transform(feats), columns=feats.columns, index=feats.index)
        errors = ae.reconstruction_errors(self.model, X) / self.model.err_scale_
        out = raw.copy()
        out["score"] = errors.max(axis=1)
        out["price_diff_pct"] = feats["price_diff_pct"]
        out["stock_diff_pct"] = feats["stock_diff_pct"]
        for i, c in enumerate(config.FEATURE_COLUMNS):
            out["err_" + c] = errors[:, i]
        out["top_feature"] = [config.FEATURE_COLUMNS[i] for i in errors.argmax(axis=1)]
        return out

    def apply_settings(self, scored, sensitivity=1.0, use_rules=None, rule_price=None, rule_sync=None, threshold=None, suppress_recovery=None):
        """Turn scores into alerts. sensitivity > 1 = more alerts, < 1 = fewer."""
        use_rules = config.USE_RULES if use_rules is None else use_rules
        rule_price = config.RULE_PRICE_DIFF if rule_price is None else rule_price
        rule_sync = config.RULE_SYNC_DELAY_MIN if rule_sync is None else rule_sync
        out = scored.copy()
        thr = (threshold or self.threshold) / sensitivity
        out["ae_flag"] = out["score"] > thr
        if config.SUPPRESS_RECOVERY if suppress_recovery is None else suppress_recovery:
            prev = out.groupby(["sku", "channel"])["ae_flag"].shift(1).fillna(False).astype(bool)
            recovery = prev & out["top_feature"].isin(["price_change_pct", "stock_change_pct"])
            out["ae_flag"] = out["ae_flag"] & ~recovery
        out["reason"] = out["top_feature"].map(config.REASONS)

        out["rule_flag"] = False
        if use_rules:
            r_price = out["price_diff_pct"].abs() > rule_price
            r_sync = out["sync_delay_min"] > rule_sync
            out["rule_flag"] = r_price | r_sync
            only_rule = out["rule_flag"] & ~out["ae_flag"]
            out.loc[only_rule & r_price, "reason"] = config.REASONS["price_diff_pct"]
            out.loc[only_rule & r_sync & ~r_price, "reason"] = config.REASONS["sync_delay_log"]

        out["flagged"] = out["ae_flag"] | out["rule_flag"]
        out["source"] = np.where(out["ae_flag"] & out["rule_flag"], "Both",
                        np.where(out["ae_flag"], "Model", np.where(out["rule_flag"], "Rule", "-")))
        ratio = out["score"] / thr
        sev = pd.Series(np.where(ratio >= config.SEVERITY_HIGH, "High",
                        np.where(ratio >= config.SEVERITY_MEDIUM, "Medium", "Low")), index=out.index)
        sev[~out["flagged"]] = "None"
        sev[out["rule_flag"] & ~out["ae_flag"]] = "Medium"   # rule-only alerts get Medium
        out["severity"] = sev
        return out

    def detect(self, raw, **settings):
        return self.apply_settings(self.score(raw), **settings)


def holdout(res):
    """The 30% test rows (same split used in training)."""
    return train_test_split(res.reset_index(drop=True), test_size=config.TEST_SIZE,
                            random_state=config.SEED)[1]


def evaluate(res, holdout_only=True):
    """Precision, recall, F1. holdout_only=True scores only the 30% test split (same split as training)."""
    if holdout_only:
        res = holdout(res)
    y, p = res["is_anomaly"], res["flagged"].astype(int)
    return (precision_score(y, p, zero_division=0), recall_score(y, p, zero_division=0),
            f1_score(y, p, zero_division=0))


if __name__ == "__main__":
    raw = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=["timestamp"])
    res = Detector().detect(raw)
    alerts = res[res["flagged"]].sort_values("score", ascending=False)
    cols = ["timestamp", "sku", "channel", "severity", "source", "reason", "score",
            "channel_price", "master_price", "channel_stock", "master_stock",
            "sync_delay_min", "is_anomaly", "anomaly_type"]
    alerts[cols].round(2).to_csv(config.ALERTS_FILE, index=False)
    print(f"Alerts: {len(alerts)} of {len(res)} rows")
    for name, hold in [("Held-out (30%)", True), ("All rows", False)]:
        p, r, f = evaluate(res, hold)
        print(f"{name}: Precision {p:.3f}  Recall {r:.3f}  F1 {f:.3f}")
    print(alerts["severity"].value_counts().to_string())
    print("Recall by type:\n", res[res.is_anomaly == 1].groupby("anomaly_type")["flagged"].mean().round(3).to_string())
