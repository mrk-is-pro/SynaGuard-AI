"""The 'Data & Model Lab' tab: one section per project milestone.
Technical, quiet, dense layout adhering to the unified design system.
"""
import numpy as np
import pandas as pd
import streamlit as st
import analysis
import charts
import config
import prepare_data
import ui
from detector import holdout
from features import FEATURE_INFO

SECTIONS = {
    "1 · Data preparation": "Inspect and clean raw sync logs; inspect normal channel baseline behavior.",
    "2 · Feature engineering": "Six engineered features extracted from multi-channel streams.",
    "3 · Autoencoder": "Self-supervised 6 → 4 → 6 compression model and architecture benchmarking.",
    "4 · Threshold analysis": "Five threshold methods evaluated on held-out validation data.",
    "5 · Accuracy & evaluation": "Model evaluation, comparison with heuristic rules, and false positive analysis.",
}


@st.cache_data
def load_features():
    return pd.read_csv(config.FEATURES_FILE)


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, theme=None, config={"displayModeBar": False})


def render(c):
    """c: dict with clean (DataFrame), quality (dict), res (scored DataFrame), det, thr, upload (bool)."""
    sec = st.radio("Section", list(SECTIONS), horizontal=True, label_visibility="collapsed")
    ui.tip(SECTIONS[sec])
    {"1": data_prep, "2": features, "3": model, "4": thresholds, "5": evaluation}[sec[0]](c)


def data_prep(c):
    q, d = c["quality"], c["clean"]
    fixed = sum(q["invalid_values_fixed"].values())

    k = st.columns(4)
    ui.kpi(k[0], "Raw logs received", f"{q['rows_in']:,}", "initial export")
    ui.kpi(k[1], "Duplicates removed", f"{q['duplicates_removed']:,}", "redundant records")
    ui.kpi(k[2], "Anomalous entries fixed", f"{fixed:,}", "negative stock / price")
    ui.kpi(k[3], "Cleaned baseline rows", f"{q['rows_out']:,}", f"{q['skus']} SKUs · {len(q['channels'])} channels")

    with st.expander("Inspection and cleaning rules log", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    ("Missing critical keys (timestamp/SKU/channel)", q["unusable_rows_dropped"], "Dropped record"),
                    ("Duplicate sync rows", q["duplicates_removed"], "Retained earliest timestamp"),
                    ("Negative / zero stock or price", fixed, "Flagged invalid and imputed"),
                    ("Missing channel price", q["missing_values_filled"]["channel_price"], "Master price × channel baseline ratio"),
                    ("Missing channel stock", q["missing_values_filled"]["channel_stock"], "Fair share of master inventory"),
                    ("Missing sync delay", q["missing_values_filled"]["sync_delay_min"], "Channel median delay"),
                    ("Temporal ordering", "-", "Chronologically sorted by SKU and channel"),
                ],
                columns=["Issue", "Occurrences", "Resolution"],
            ).astype({"Occurrences": str}),
            hide_index=True,
            width="stretch",
        )

    ui.section("Channel baseline statistics", "Observed pricing and inventory distribution across active channels.")
    st.dataframe(prepare_data.channel_stats(d), width="stretch")

    ui.section("Normal operating distribution", "Baseline distributions across price, stock, sync latency, and velocity.")
    normal = d[d["is_anomaly"] == 0] if "is_anomaly" in d.columns else d
    t1, t2, t3, t4 = st.tabs(["Price ratio spread", "Inventory vs fair share", "Sync latency", "Sales velocity"])
    with t1:
        st.caption("Distribution of channel price relative to master catalog price.")
        show(charts.price_ratio_box(normal), "dp_price")
    with t2:
        st.caption("Channel stock against fair share of master inventory. Deviations indicate sync drift.")
        show(charts.stock_scatter(normal), "dp_stock")
    with t3:
        st.caption("Sync delay frequency. Long delays indicate stalled pipelines.")
        show(charts.delay_hist(normal), "dp_delay")
    with t4:
        st.caption("Hourly units sold distribution across channels.")
        show(charts.units_hist(normal), "dp_units")

    with st.expander("Cleaned dataset sample & export"):
        st.dataframe(d.head(100), hide_index=True, width="stretch", height=220)
        st.download_button("⬇️ Download cleaned CSV", d.to_csv(index=False), "clean_sync_data.csv")


def features(c):
    ui.section("Engineered feature set", "Features computed from raw sync streams for autoencoder consumption.")
    st.dataframe(FEATURE_INFO, hide_index=True, width="stretch")
    st.caption("Standardized to zero mean and unit variance. Unsupervised learning: error labels are never provided.")

    f = load_features()
    t1, t2 = st.tabs(["Feature distributions", "Feature correlations"])
    with t1:
        st.caption("Feature separation between normal records and real injected errors.")
        show(charts.feature_dist(f), "fe_dist")
    with t2:
        st.caption("Inter-feature correlation matrix.")
        show(charts.corr_heat(f), "fe_corr")


def model(c):
    det, m, e = c["det"], c["art"]["metrics"], c["art"]["experiments"]
    ui.section("Autoencoder architecture", "Bottleneck network trained to reconstruct normal synchronization states.")
    st.caption(
        f"Input ({len(config.FEATURE_COLUMNS)}) → Hidden ({' → '.join(map(str, config.HIDDEN_LAYERS))}) → "
        f"Output ({len(config.FEATURE_COLUMNS)}). Activation: tanh. Optimization: Adam on scikit-learn MLPRegressor."
    )

    k = st.columns(4)
    ui.kpi(k[0], "Training volume", f"{m['n_train']:,}", "70% split")
    ui.kpi(k[1], "Unsupervised trim", f"{m['trimmed_rows']:,}", "2% worst first-pass errors")
    ui.kpi(k[2], "Iterations to convergence", f"{m['iterations']}", "early stopping")
    ui.kpi(k[3], "Final reconstruction loss", f"{m['final_loss']:.4f}", "MSE on clean training set")

    t1, t2, t3 = st.tabs(["Training loss curve", "Architecture benchmark", "Feature reconstruction demo"])
    with t1:
        st.caption("Loss convergence during second-pass model training.")
        show(charts.loss_curve(det.model), "md_loss")
    with t2:
        if e is None:
            st.info("Run `python experiments.py` to compare architectures.")
        else:
            st.caption("Comparative held-out F1 across candidate hidden layer configurations.")
            show(charts.arch_bars(e), "md_arch")
            st.dataframe(
                e.rename(columns={"normal_mse": "Error on normal rows", "seconds": "Train time (s)"})
                .drop(columns=["hidden", "chosen"])
                .round(3),
                hide_index=True,
                width="stretch",
            )
    with t3:
        f = load_features()
        kinds = ["A normal row"] + sorted(f.loc[f["is_anomaly"] == 1, "anomaly_type"].unique())
        c1, c2 = st.columns([2, 1])
        pick = c1.selectbox("Select error category", kinds)
        seed = c2.number_input("Sample index", 0, 50, 0)
        sub = f[f["is_anomaly"] == 0] if pick == "A normal row" else f[f["anomaly_type"] == pick]
        row = sub.iloc[int(seed) % len(sub)]
        x = row[config.FEATURE_COLUMNS].values.astype(float)
        r = det.model.predict(x.reshape(1, -1))[0]
        st.caption("Reconstruction comparison: large differences indicate anomalous input dimensions.")
        show(charts.reconstruction(np.clip(x, -6, 6), np.clip(r, -6, 6), pick), "md_recon")


def thresholds(c):
    art, h = c["art"], c["hold"]
    t = art["thresholds"]
    ev = pd.DataFrame(t["evaluation"])

    ui.section("Threshold selection methods", "Comparing statistical and empirical cutoff strategies.")
    st.caption(
        f"Active method: **{ev.loc[ev['method'] == t['chosen_method'], 'name'].iloc[0]}** (threshold value: {t['chosen']:.3f})"
    )

    show_df = ev[["name", "threshold", "alerts", "precision", "recall", "f1"]].rename(
        columns={
            "name": "Method",
            "threshold": "Threshold cutoff",
            "alerts": "Alerts flagged",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
        }
    )
    st.dataframe(show_df.round(3), hide_index=True, width="stretch")

    if h is None:
        return

    t1, t2, t3 = st.tabs(["Score distribution & cutoffs", "Method metric comparison", "Precision-Recall curve"])
    with t1:
        st.caption("Dashed lines indicate candidate thresholds against normal vs anomalous score densities.")
        show(charts.thr_hist(h, t["methods"]), "th_hist")
    with t2:
        st.caption("Precision, recall, and F1 across each threshold calculation method.")
        show(charts.method_bars(ev), "th_bars")
    with t3:
        st.caption("Precision-Recall trade-off curve across operating sensitivities.")
        show(charts.pr_curve(h["is_anomaly"], h["score"], c["thr"]), "th_pr")


def evaluation(c):
    h = c["hold"]
    if h is None:
        st.info("Uploaded data does not contain validation labels (`is_anomaly`).")
        return

    comp = analysis.comparison_table(h)
    best = comp.iloc[2]

    k = st.columns(4)
    ui.kpi(k[0], "Precision", f"{best['Precision']:.1%}", "true positive rate")
    ui.kpi(k[1], "Recall", f"{best['Recall']:.1%}", "detection coverage")
    ui.kpi(k[2], "F1 Score", f"{best['F1']:.1%}", "balanced harmonic mean")
    ui.kpi(k[3], "Missed anomalies", f"{int(((h['is_anomaly'] == 1) & ~h['flagged']).sum())}", f"of {int(h['is_anomaly'].sum()):,} real errors")

    ui.section("Model vs heuristic rules", "Benchmarking unsupervised autoencoder against traditional monitoring rules.")
    st.dataframe(
        comp.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )
    show(charts.approach_bars(analysis.type_comparison(h)), "ev_types")

    fa = analysis.fp_analysis(h, c["thr"])
    ui.section("False alarm analysis", "Understanding non-error deviations flagged by the model.")
    for line in fa["summary"]:
        st.markdown(f"• {line}")

    with st.expander("Recovery suppression effect", expanded=False):
        st.caption("Suppressing alerts triggered solely by post-correction recovery jumps.")
        rec = analysis.recovery_effect(c["det"], c["scored"], c["settings"])
        st.dataframe(
            rec.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )

    t1, t2, t3, t4 = st.tabs(["Trigger frequency", "Group feature profile", "Confusion & score density", "Audit samples"])
    with t1:
        if len(fa["fp"]):
            a, b = st.columns(2)
            with a:
                st.markdown("**False alarms by trigger**")
                show(charts.reasons_bar(fa["reasons"]), "ev_reasons")
            with b:
                st.markdown("**False alarms by channel**")
                show(charts.reasons_bar(fa["by_channel"]), "ev_ch")
    with t2:
        st.caption("Mean relative reconstruction error by group.")
        show(charts.group_heat(fa["groups"]), "ev_group")
    with t3:
        a, b = st.columns(2)
        with a:
            st.markdown("**Confusion matrix**")
            show(charts.confusion(h["is_anomaly"], h["flagged"].astype(int)), "ev_conf")
        with b:
            st.markdown("**Score distribution**")
            show(charts.score_hist(h, c["thr"]), "ev_hist")
    with t4:
        cols = ["timestamp", "sku", "channel", "reason", "source", "score"]
        st.markdown("**Sample false alarms**")
        st.dataframe(fa["fp"].sort_values("score", ascending=False)[cols].head(10).round(2), hide_index=True, width="stretch")
        st.markdown("**Sample false negatives (missed errors)**")
        st.dataframe(fa["fn"][cols[:3] + ["anomaly_type", "score"]].head(10).round(2), hide_index=True, width="stretch")
