"""Turns raw alerts into business insights: likely type, money at risk, suggested action."""
import numpy as np
import pandas as pd
import config

ACTIONS = {
    "Decimal / entry error": "Pause the listing on this channel and re-push the master price immediately.",
    "Price mismatch": "Check the channel's price rules and the last price-feed job; re-sync the price.",
    "Stale sync": "Check the integration job / API limits; trigger a manual sync.",
    "Oversell risk": "Lower or zero the channel's available stock and re-sync inventory now.",
    "Stock drift": "Reconcile channel stock against the ERP and fix the allocation rule.",
    "Unusual pattern": "Review the SKU manually; compare recent syncs for this channel.",
}


def enrich(df):
    """Add likely_type, est_impact and action to a table of alert rows."""
    d = df.copy()
    if d.empty:
        for c in ["likely_type", "est_impact", "action"]:
            d[c] = []
        return d
    p, s = d["price_diff_pct"], d["stock_diff_pct"]
    conds = [p < -0.8, p.abs() > config.RULE_PRICE_DIFF, d["sync_delay_min"] > config.RULE_SYNC_DELAY_MIN,
             d["channel_stock"] <= 5, s.abs() > 0.25]
    d["likely_type"] = np.select(conds, list(ACTIONS)[:5], "Unusual pattern")
    units = d["units_sold"].clip(lower=1) * config.EXPOSURE_HOURS
    gap = p.abs() * d["master_price"]
    impact = np.select(
        conds,
        [gap * units, gap * units, units * d["master_price"] * 0.05,
         units * d["master_price"] * config.OVERSELL_COST_PCT,
         (d["channel_stock"] - d["master_stock"] / len(config.CHANNELS)).abs() * d["master_price"] * 0.1],
        default=units * d["master_price"] * 0.02)
    d["est_impact"] = np.round(impact, 2)
    d["action"] = d["likely_type"].map(ACTIONS)
    ts = pd.to_datetime(d["timestamp"])
    d["alert_id"] = d["sku"].astype(str) + "|" + d["channel"].astype(str) + "|" + ts.dt.strftime("%Y%m%d%H")
    return d
