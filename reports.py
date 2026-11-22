"""Builds the two project deliverables as self-contained HTML reports (open in a browser; Print > Save as PDF).
Run:  python reports.py"""
import json
import os
from datetime import datetime
import pandas as pd
import analysis
import charts
import config
import prepare_data
import ui
from detector import Detector, holdout
from features import FEATURE_INFO
from insights import ACTIONS

CSS = """<style>
body{font-family:'Segoe UI',Arial,sans-serif;color:#2B2340;background:#FCF8F3;margin:0;line-height:1.55}
.page{max-width:980px;margin:0 auto;padding:30px 28px 60px}
.cover{background:linear-gradient(120deg,#FFD9CC,#E9E1FF);border:1px solid #E8E1F0;border-radius:18px;padding:26px 30px;display:flex;gap:20px;align-items:center}
.cover h1{margin:0;color:#2B2340;font-size:2rem}.cover p{margin:4px 0 0;color:#6E6685}
h2{color:#2B2340;border-bottom:2px solid #E8E1F0;padding-bottom:4px;margin-top:38px}h3{color:#6E6685}
table.t{border-collapse:collapse;width:100%;margin:10px 0;font-size:.92rem;background:#fff}
table.t th{background:#F3EEFB;text-align:left;padding:7px 10px;border:1px solid #E8E1F0}
table.t td{padding:6px 10px;border:1px solid #E8E1F0}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}.k{flex:1;min-width:150px;background:#fff;border:1px solid #E8E1F0;border-top:5px solid #F26A4B;border-radius:14px;padding:10px 14px}
.k b{display:block;font-size:1.5rem}.k span{color:#6E6685;font-size:.85rem}
.note{background:#F3EEFB;border:1px solid #D9CCFA;border-radius:10px;padding:8px 14px;margin:10px 0}
code,pre{background:#F3EEFB;border-radius:6px;padding:1px 6px;font-size:.88rem}pre{padding:10px;overflow:auto}
.fig{background:#fff;border:1px solid #E8E1F0;border-radius:14px;padding:8px;margin:12px 0}
@media print{body{background:#fff}.page{padding:0}}
</style>"""


class Doc:
    def __init__(self, title, subtitle):
        self.parts, self.first = [], True
        self.title = title
        self.parts.append(f'<div class="cover">{ui.logo_html(72)}<div><h1>{title}</h1><p>{subtitle}</p>'
                          f'<p>Team 14 · generated {datetime.now():%B %d, %Y}</p></div></div>')

    def h(self, t): self.parts.append(f"<h2>{t}</h2>")
    def h3(self, t): self.parts.append(f"<h3>{t}</h3>")
    def p(self, t): self.parts.append(f"<p>{t}</p>")
    def note(self, t): self.parts.append(f'<div class="note">{t}</div>')
    def ul(self, items): self.parts.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
    def code(self, t): self.parts.append(f"<pre>{t}</pre>")
    def kpis(self, items):
        self.parts.append('<div class="kpis">' + "".join(f'<div class="k"><b>{v}</b><span>{l}</span></div>' for l, v in items) + "</div>")

    def table(self, df, index=False, fmt=None):
        self.parts.append(df.to_html(classes="t", index=index, border=0, float_format=(fmt or (lambda x: f"{x:,.3f}")), escape=False))

    def fig(self, f):
        f.update_layout(template="plotly_white")
        html = f.to_html(full_html=False, include_plotlyjs="cdn" if self.first else False, config={"displayModeBar": False})
        self.first = False
        self.parts.append(f'<div class="fig">{html}</div>')

    def save(self, name):
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        path = os.path.join(config.REPORTS_DIR, name)
        open(path, "w", encoding="utf-8").write(
            f"<!doctype html><html><head><meta charset='utf-8'><title>{self.title}</title>{CSS}</head>"
            f"<body><div class='page'>{''.join(self.parts)}</div></body></html>")
        return path


COLUMNS = pd.DataFrame([
    ("timestamp", "When the sync happened"), ("sku", "Product code"), ("channel", "Sales channel (Shopify, Amazon, eBay)"),
    ("master_price / channel_price", "Price in the master system vs shown on the channel"),
    ("master_stock / channel_stock", "Stock in the master system vs shown on the channel"),
    ("units_sold", "Units sold in that hour"), ("sync_delay_min", "Minutes since the channel was last updated"),
    ("sync_attempts / sync_status", "Attempts the sync needed, and the status the integration reported (success, retry, failed)"),
    ("is_anomaly / anomaly_type", "Injected-error labels, used only for evaluation, never for training")], columns=["Column", "Meaning"])


def pct(x): return f"{x:.1%}"


def report_data_prep(clean_df, q, feats):
    d = Doc("Pricing & Inventory Dataset Preparation Report", "SyncGuard AI · Week 1 deliverable")
    normal = clean_df[clean_df["is_anomaly"] == 0]
    d.h("1. Objective")
    d.p("Prepare a documented, clean baseline of pricing and inventory synchronization records so an autoencoder can learn what "
        "normal syncing looks like. This report covers the dataset, the problems found, how each was fixed, channel-level statistics, "
        "visualizations of normal behavior, and the engineered features.")
    d.h("2. Dataset")
    d.p(f"No public dataset of master-vs-channel sync logs exists, so a realistic synthetic dataset was generated "
        f"(<code>data_generator.py</code>). It simulates {q['skus']} products sold on {len(q['channels'])} channels "
        f"({', '.join(q['channels'])}) with hourly sync records from {q['date_from'][:10]} to {q['date_to'][:10]}. "
        f"A small share of rows ({config.ANOMALY_RATE:.0%}) has injected errors (price mismatch, decimal slip, stale sync, stock drift, "
        "oversell risk) so detection quality can be measured. Legitimate-but-unusual behaviour is included too: promotions (the master price drops and each channel catches up an hour later), bulk inventory deliveries and occasional sync retries. Operational messiness (duplicates, gaps, impossible values) was also added "
        "to mimic a real export. A real dataset can be dropped in without code changes beyond the loader.")
    d.table(COLUMNS)
    d.kpis([("Raw rows", f"{q['rows_in']:,}"), ("Clean rows", f"{q['rows_out']:,}"), ("Products", q["skus"]), ("Channels", len(q["channels"]))])
    d.h("3. Inspection and cleaning")
    fixed = sum(q["invalid_values_fixed"].values())
    d.table(pd.DataFrame([
        ("Rows missing timestamp, product, channel or master values", q["unusable_rows_dropped"], "Dropped: cannot be interpreted"),
        ("Duplicate rows (same product, channel, time)", q["duplicates_removed"], "Kept first copy"),
        ("Negative / zero stock or price", fixed, "Treated as missing, then imputed"),
        ("Missing channel price", q["missing_values_filled"]["channel_price"], "Master price × the channel's normal ratio"),
        ("Missing channel stock", q["missing_values_filled"]["channel_stock"], "Fair share of master stock"),
        ("Missing sync delay", q["missing_values_filled"]["sync_delay_min"], "Median delay of that channel"),
        ("Row order", "-", "Sorted by product, channel, time (needed for change features)")],
        columns=["Issue found", "Rows", "Treatment"]).astype({"Rows": str}))
    d.note("Imputation uses values that look normal, so a repaired row is never mistaken for an anomaly just because it was damaged. "
           "The trade-off is that a real error sitting in a damaged row is lost; this affects well under 1% of rows.")
    d.h("4. Channel-level statistics")
    d.table(prepare_data.channel_stats(clean_df), index=True)
    d.p("Each channel has its own stable pricing ratio to the master (for example Amazon about +3%, eBay about −2%), which is why the "
        "model learns a per-channel baseline instead of using one global rule.")
    d.h("4b. Encoding categorical and timestamp attributes")
    d.table(pd.DataFrame([("sku, channel", "Category codes and one-hot channel columns"), ("sync_status", "success = 0, retry = 1, failed = 2"),
                          ("timestamp", "Hour of day and day of week as sine/cosine pairs (23:00 sits next to 00:00), plus epoch seconds"),
                          ("update interval", "Minutes since the previous update of the same product and channel, log-scaled")], columns=["Attribute", "Numeric representation"]))
    d.p(f"The fully encoded table is saved as <code>{config.ENCODED_FILE}</code>. The model learns from the engineered features below, standardized (mean 0, spread 1) so no feature dominates.")
    d.h("5. Normal pricing and inventory behavior")
    d.fig(charts.price_ratio_box(normal)); d.p("Price ratio by channel: tight, stable bands.")
    d.fig(charts.stock_scatter(normal)); d.p("Channel stock tracks its fair share of master stock.")
    d.fig(charts.delay_hist(normal)); d.p("Sync delay: most syncs complete within minutes, with a thin tail.")
    d.fig(charts.units_hist(normal)); d.p("Hourly sales volume.")
    d.h("6. Feature engineering")
    d.table(FEATURE_INFO)
    d.fig(charts.feature_dist(feats)); d.p("Scaled feature distributions: normal rows (grey) versus injected errors (red).")
    d.fig(charts.corr_heat(feats)); d.p("Correlations between features.")
    d.h("7. Output")
    d.p(f"A cleaned file (<code>{config.CLEAN_DATA_FILE}</code>), a learned per-channel baseline, a fitted scaler and a scaled feature table "
        "form the baseline dataset handed to the modeling stage. The encoded dataset is saved alongside it.")
    return d.save("01_data_preparation_report.html")


def report_final(clean_df, q, feats, res, hold, det, art, scored):
    m, t, exp = art["metrics"], art["thresholds"], art["experiments"]
    comp, types = analysis.comparison_table(hold), analysis.type_comparison(hold)
    fa = analysis.fp_analysis(hold, det.threshold)
    rec = analysis.recovery_effect(det, scored, {})
    best = comp.iloc[2]
    d = Doc("SyncGuard AI: Final Project Report", "Pricing & Inventory Sync Anomaly Detector")
    d.h("1. Executive summary")
    d.p("SyncGuard AI learns what normal price and inventory syncing looks like using an unsupervised autoencoder and flags syncs that "
        "deviate, so operations teams can fix integration problems before customers see them. It is delivered as a data pipeline, a model, "
        "a FastAPI detection service and an interactive dashboard.")
    d.kpis([("Held-out precision", pct(best["Precision"])), ("Held-out recall", pct(best["Recall"])), ("Held-out F1", pct(best["F1"])),
            ("ROC-AUC", f"{m['roc_auc']:.3f}"), ("PR-AUC", f"{m['pr_auc']:.3f}")])
    d.h("2. Problem")
    d.p("Retailers selling on several channels depend on automated syncing of prices, stock and availability. Failures create stale prices, wrong "
        "stock and overselling. Fixed-threshold monitoring only catches what someone thought to configure, and misses combined or subtle problems. "
        "An autoencoder learns normal behavior from data and flags anything it cannot reproduce.")
    d.h("3. Data pipeline")
    d.p(f"{q['rows_in']:,} raw rows were cleaned to {q['rows_out']:,} ({q['duplicates_removed']:,} duplicates removed, "
        f"{sum(q['missing_values_filled'].values()):,} gaps filled, {sum(q['invalid_values_fixed'].values()):,} impossible values fixed). "
        "Full details are in the Dataset Preparation Report.")
    d.table(prepare_data.channel_stats(clean_df), index=True)
    d.h("4. Feature engineering")
    d.table(FEATURE_INFO)
    d.h("5. Autoencoder model")
    d.p(f"A scikit-learn <code>MLPRegressor</code> with architecture {len(config.FEATURE_COLUMNS)} → {' → '.join(map(str, config.HIDDEN_LAYERS))} → "
        f"{len(config.FEATURE_COLUMNS)} (tanh) is trained to reproduce its input. Training is label-free and two-pass: a first model scores the "
        f"training rows, the most unusual {config.TRIM_PERCENT}% ({m['trimmed_rows']:,} rows) are dropped as probable errors, and the model is retrained "
        f"on the cleaned set. Training used {m['n_train']:,} rows and {m['iterations']} iterations.")
    d.fig(charts.loss_curve(det.model))
    if exp is not None:
        d.h3("Architecture experiments")
        d.fig(charts.arch_bars(exp))
        d.table(exp.drop(columns=["hidden", "chosen"]))
        comp_ = exp[~exp["architecture"].str.contains("no bottleneck")]
        bc, nb = comp_.sort_values("f1", ascending=False).iloc[0], exp[exp["architecture"].str.contains("no bottleneck")]
        d.p(f"Among designs that compress the input, {bc['architecture'].strip()} scored best (F1 {bc['f1']:.2f}). " +
            (f"The design without any bottleneck scored F1 {nb.iloc[0]['f1']:.2f}; with no compression the network can simply copy its input, so it is only a baseline. " if len(nb) else "") +
            "Deeper networks overfit the normal rows and raised more false alarms. The experiment used a 20,000-row sample for speed.")
    d.h("6. Anomaly scoring")
    d.p("For each row the model's reconstruction error is computed per feature, divided by that feature's normal error range, and the worst "
        "feature gives the row's anomaly score. This stops one large mistake from being averaged away by five normal features, and the worst "
        "feature also provides a plain-English reason. Severity is the score relative to the threshold "
        f"(High ≥ {config.SEVERITY_HIGH}×, Medium ≥ {config.SEVERITY_MEDIUM}×).")
    d.h("7. Threshold analysis")
    ev = pd.DataFrame(t["evaluation"])[["name", "threshold", "alerts", "precision", "recall", "f1"]]
    d.table(ev.rename(columns={"name": "Method"}))
    d.fig(charts.thr_hist(hold, t["methods"]))
    d.fig(charts.pr_curve(hold["is_anomaly"], hold["score"], det.threshold))
    d.p(f"The live threshold uses the {config.THRESHOLD_PERCENTILE}th percentile of scores on the cleaned training rows. Methods assuming a "
        "bell-shaped score (mean + 3σ) or using very tight robust spreads (MAD, IQR) were poorly matched to these heavy-tailed scores: "
        "they either missed many errors or raised thousands of alerts. A sensitivity control and an alternative-method selector let operators move along the curve.")
    d.h("8. Detection workflow and FastAPI service")
    d.p("The trained model is exposed through a FastAPI service (<code>api.py</code>) that validates input with Pydantic and returns the score, "
        "severity, reason, likely error type, estimated revenue at risk and suggested action. The same scoring code serves the dashboard.")
    d.table(pd.DataFrame([("GET", "/health", "Service and model status"), ("GET", "/model/info", "Model design and threshold"),
                          ("GET", "/thresholds", "All threshold methods and performance"), ("GET", "/metrics", "Held-out evaluation"),
                          ("POST", "/score", "Score one record"), ("POST", "/score/batch", "Score up to 5,000 records")],
                         columns=["Method", "Path", "Purpose"]))
    d.code("uvicorn api:app --reload      # then open http://127.0.0.1:8000/docs\npython test_api.py            # automated API checks")
    d.p("Automated tests check health, normal records, a decimal-error record, a stale sync, input validation and batch scoring.")
    d.h("9. Evaluation")
    d.p(f"Evaluated on the held-out {config.TEST_SIZE:.0%} of the data, which the model never saw during training.")
    d.table(comp.assign(Precision=comp.Precision.map(pct), Recall=comp.Recall.map(pct), F1=comp.F1.map(pct)))
    d.fig(charts.approach_bars(types))
    d.p("Rules only look at price gaps and sync age, so they cannot see stock problems. The autoencoder covers all six signals; the hybrid "
        "adds the rules as a safety net for obvious cases.")
    d.fig(charts.confusion(hold["is_anomaly"], hold["flagged"].astype(int)))
    d.h("10. False-positive and error analysis")
    d.ul(fa["summary"])
    d.fig(charts.group_heat(fa["groups"]))
    d.p("False alarms and real errors have different signatures. A recurring pattern is the record right after a real error: when a price or "
        "stock count snaps back to normal it looks like a sudden change. The system therefore skips alerts triggered only by a sudden change "
        "in the sync right after an alert (recovery suppression):")
    d.table(rec.assign(Precision=rec.Precision.map(pct), Recall=rec.Recall.map(pct), F1=rec.F1.map(pct)))
    d.h("10.1 Causes of false alarms")
    causes = analysis.fp_causes(hold)
    d.table(causes.rename_axis("Cause").reset_index(name="False alarms"))
    d.p("Promotions create a short, legitimate disagreement: the master price drops first and each channel catches up an hour later. Bulk inventory deliveries move master and channel stock together, so they cancel out in the “change versus master” features and rarely cause alarms. Seasonal demand surges and unusual-but-valid channel activity are not simulated and are listed as limitations.")
    d.h("10.2 Cross-channel inconsistencies")
    cc = analysis.cross_channel(res)
    d.fig(charts.spread_bars(cc))
    d.table(cc.round(3), index=True)
    d.p("When the same product disagrees strongly across channels, the share of rows flagged jumps from about 1% to roughly a quarter, and the share that are real errors rises the same way: the scores reflect cross-channel inconsistency.")
    d.h("10.3 Highest-scoring events")
    top_, bytype = analysis.suspicious(res)
    d.table(top_.round(2))
    d.p("Across all alerts the most common likely causes are: " + ", ".join(f"{k} ({v})" for k, v in bytype.head(4).items()) + ".")
    d.h("11. Limitations")
    d.p("Where the model may be unreliable, with the evidence from this project:")
    hist = art.get("history")
    d.ul(["<b>Insufficient history.</b> " + (f"With 1 day of training data F1 was {hist.iloc[0]['f1']:.2f}; with {int(hist.iloc[-1]['days_of_history'])} days it was {hist.iloc[-1]['f1']:.2f} (table below)." if hist is not None else "Less history means a poorer picture of normal."),
          "<b>New products and new sales channels.</b> With no history there is no baseline price ratio and no normal pattern, so early scores are unreliable until enough syncs accumulate.",
          "<b>Promotional campaigns.</b> A promotion makes channels briefly disagree with the master while they catch up; these appear as false alarms (section 10.1).",
          "<b>Bulk inventory updates.</b> Handled well when master and channel move together, but a one-sided bulk change would look like stock drift.",
          "<b>Sudden demand changes.</b> A genuine sales surge lowers stock cover quickly and can look like oversell risk. Not simulated here.",
          "<b>Synthetic data.</b> Results show the method&rsquo;s behaviour, not real-world performance. Mild errors overlap with normal noise.",
          "<b>Fixed threshold.</b> Normal behaviour that drifts needs retraining. Single-record API calls cannot see sudden changes (no history).",
          "Revenue at risk is an illustrative estimate based on stated assumptions."])
    if hist is not None:
        d.fig(charts.history_lines(hist)); d.table(hist.round(3))
    d.h("12. Operational recommendations")
    d.ul(["Start with the Balanced alert level and review High alerts first; tune sensitivity using the false-alarm rate your team can handle.",
          "Run batch scoring after each sync cycle or call the API from the integration job.",
          "Record analyst decisions (the dashboard's status workflow) and use confirmed false positives to tune thresholds.",
          "Retrain monthly or after major catalog, promotion or channel changes.",
          "Keep the safety rules on for business-critical limits (very large price gaps, very old syncs).",
          "Keep recovery suppression on so analysts are not alerted twice for one incident."])
    d.table(pd.DataFrame({"Error type": list(ACTIONS), "Suggested response": list(ACTIONS.values())}))
    d.h("13. Conclusions and future work")
    d.p(f"An unsupervised autoencoder learned normal sync behavior without labels and, with a tuned threshold, found {pct(comp.iloc[1]['Recall'])} "
        f"of real errors at {pct(comp.iloc[1]['Precision'])} precision on its own, and {pct(best['Recall'])} recall when combined with simple rules. "
        "Future work: validate on real operational logs, add rolling-window and cross-channel features, per-product thresholds, "
        "sequence models, streaming ingestion and alert delivery to chat or email.")
    return d.save("02_final_project_report.html")


if __name__ == "__main__":
    clean_df = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=["timestamp"])
    feats = pd.read_csv(config.FEATURES_FILE)
    art = analysis.artifacts()
    det = Detector()
    res = analysis.add_context(det.detect(clean_df))
    hold = holdout(res)
    print("Wrote", report_data_prep(clean_df, art["quality"], feats))
    print("Wrote", report_final(clean_df, art["quality"], feats, res, hold, det, art, det.score(clean_df)))
