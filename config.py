"""All settings live here. Change a value here, nothing else needs editing."""
import os

# ---- Data generation ----
SEED = 42                      # makes results repeatable
N_SKUS = 30                    # number of products
CHANNELS = {                   # channel name -> normal price markup vs master
    "Shopify": 0.00,
    "Amazon": 0.03,
    "eBay": -0.02,
}
DAYS = 30                      # days of history
RECORDS_PER_DAY = 24           # one sync record per hour
ANOMALY_RATE = 0.02            # 2% of rows get an injected error

# ---- Files ----
RAW_DATA_FILE = "data/raw/raw_sync_data.csv"      # as exported (messy)
FEATURES_FILE = "data/processed/features.csv"
SCALER_FILE = "models/scaler.joblib"

# ---- Features fed to the model (Phase 2 reads this list) ----
FEATURE_COLUMNS = [
    "price_diff_pct",      # channel price vs expected price
    "stock_diff_pct",      # channel stock vs master stock share
    "sync_delay_log",      # log of minutes since last sync
    "stock_cover_hours",   # stock / sales rate (low = oversell risk)
    "price_change_pct",    # price jump vs previous record
    "stock_change_pct",    # stock jump vs previous record
    "sync_attempts_log",   # log of how many attempts the sync needed
    "update_gap_log",      # log of minutes since the previous update (update interval)
    "sync_status_code",    # 0 success, 1 retry, 2 failed
]

# ---- Phase 2: model ----
MODEL_FILE = "models/autoencoder.joblib"
THRESHOLD_FILE = "models/threshold.json"
HIDDEN_LAYERS = (4,)           # one narrow layer (9 inputs -> 4 -> 9); chosen by experiments.py
MAX_ITER = 200
TEST_SIZE = 0.3                # 30% of rows held out for evaluation
THRESHOLD_PERCENTILE = 99.6      # % of cleaned training rows treated as normal; raise = fewer alerts
ERROR_SCALE_PERCENTILE = 99    # each feature's error is judged relative to its own normal range

# ---- Phase 3: detection engine ----
ALERTS_FILE = "results/alerts.csv"
USE_RULES = True               # add simple safety-net rules on top of the autoencoder
RULE_PRICE_DIFF = 0.15         # flag if channel price is >15% off its normal
RULE_SYNC_DELAY_MIN = 60       # flag if last sync was >60 minutes ago
SEVERITY_HIGH = 150            # score / threshold >= 150 -> High (scores span orders of magnitude)
SEVERITY_MEDIUM = 20           # score / threshold >= 20 -> Medium, else Low
REASONS = {                    # feature -> plain-English explanation
    "price_diff_pct": "Channel price differs from the master price",
    "stock_diff_pct": "Channel stock doesn't match master stock",
    "sync_delay_log": "Sync is late / stale",
    "stock_cover_hours": "Stock is very low versus sales speed (oversell risk)",
    "price_change_pct": "Price changed suddenly since last sync",
    "stock_change_pct": "Stock changed suddenly since last sync",
    "sync_attempts_log": "Sync needed an unusual number of attempts",
    "update_gap_log": "Updates arrived at an unusual interval",
    "sync_status_code": "Sync reported a retry or failure",
}

# ---- Improvements ----
TRIM_PERCENT = 4               # retrain after dropping the 4% most unusual training rows (label-free cleaning)

# ---- Business-impact estimate (illustrative assumptions, shown in the dashboard) ----
CURRENCY = "$"
EXPOSURE_HOURS = 6             # assumed hours an error stays live before someone fixes it
OVERSELL_COST_PCT = 0.5        # assumed cost of an oversold unit (refund, penalty) as % of price

# ---- Data preparation, thresholds, API, reports ----
CLEAN_DATA_FILE = "data/processed/sync_data.csv"        # cleaned data used by everything else
QUALITY_FILE = "data/processed/data_quality.json"
BASELINE_FILE = "models/baseline.json"        # normal price ratio per channel (learned from data)
THRESHOLDS_FILE = "results/thresholds.json"    # all threshold methods + the chosen one
METRICS_FILE = "results/metrics.json"
EXPERIMENTS_FILE = "results/experiments.csv"
REPORTS_DIR = "reports"
MESSY_RATE = 0.004                            # share of rows given duplicates / missing / negative values
THRESHOLD_METHOD = "percentile"               # which method sets the live alert threshold
SUPPRESS_RECOVERY = True       # don't re-alert on the sync right after an alert (price/stock snapping back looks like a "sudden change")

# ---- Project structure (WBS 1.1.2) ----
ENCODED_FILE = "data/processed/encoded_sync_data.csv"   # categorical + timestamp attributes as numbers
HISTORY_FILE = "results/history_experiment.csv"         # how much history the model needs
PROMO_EVENTS = 10              # legitimate promotions injected (valid but unusual)
BULK_EVENTS = 15               # legitimate bulk inventory updates injected
DIRS = ["data/raw", "data/processed", "models", "results", "reports", "notebooks", "docs", "api", "tests"]
for _d in DIRS:
    os.makedirs(_d, exist_ok=True)
