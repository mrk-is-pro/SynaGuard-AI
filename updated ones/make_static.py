"""Collect the Python study results (held-out comparison, thresholds, designs, false alarms, AUC, history) for the dashboard."""
import json
import os
import sys
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import analysis  # noqa: E402
import config  # noqa: E402
from detector import Detector, holdout  # noqa: E402

det = Detector()
clean = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=["timestamp"])
scored = det.score(clean)
res = analysis.add_context(det.apply_settings(scored))
h = holdout(res)
art = analysis.artifacts()
rows = lambda df: json.loads(df.round(3).to_json(orient="records"))
ev = pd.DataFrame(art["thresholds"]["evaluation"])[["name", "threshold", "alerts", "precision", "recall", "f1"]]
out = {"cmp": rows(analysis.comparison_table(h)), "types": json.loads(analysis.type_comparison(h).round(3).to_json()),
       "thr": rows(ev), "arch": rows(art["experiments"][["architecture", "f1", "recall", "precision", "chosen"]]),
       "fp": analysis.fp_analysis(h, det.threshold)["summary"], "rec": rows(analysis.recovery_effect(det, scored, {})),
       "met": {k: art["metrics"][k] for k in ["roc_auc", "pr_auc"]},
       "hist": rows(art["history"]) if art["history"] is not None else []}
dst = os.path.join(os.path.dirname(__file__), "generated", "static.json")
open(dst, "w").write(json.dumps(out, separators=(",", ":")))
print("static.json", os.path.getsize(dst) // 1024, "KB")
