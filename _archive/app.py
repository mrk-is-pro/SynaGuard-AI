"""SyncGuard AI: Mission Control & Cyber Telemetry Dashboard.
Obsidian Dark Theme with Glowing Neon Accents and Vertical Sidebar Navigation.
Run with: streamlit run app.py
"""
import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

import analysis
import charts
import config
import lab
import service
import store
import thresholds as thr_mod
import ui
from detector import Detector, holdout
from insights import ACTIONS, enrich
from prepare_data import clean

st.set_page_config(
    page_title="SyncGuard // AI Mission Control",
    page_icon=ui.page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

if not (os.path.exists(config.MODEL_FILE) and os.path.exists(config.CLEAN_DATA_FILE)):
    st.error("⚠️ Model or data artifacts not found. Execute `python run_all.py` first.")
    st.stop()

REQUIRED = [
    "timestamp",
    "sku",
    "channel",
    "master_price",
    "channel_price",
    "master_stock",
    "channel_stock",
    "units_sold",
    "sync_delay_min",
]
URGENCY = {"High": "Urgent", "Medium": "Soon", "Low": "Low"}
TYPE_MAP = {
    "decimal_error": "Decimal / entry error",
    "price_mismatch": "Price mismatch",
    "stale_sync": "Stale sync",
    "oversell_risk": "Oversell risk",
    "stock_drift": "Stock drift",
}
LEVELS = {"Fewer alerts": 0.5, "Balanced": 1.0, "More alerts": 2.0}
CUR = config.CURRENCY


@st.cache_resource
def get_detector():
    return Detector()


@st.cache_data
def score_data(raw):
    return get_detector().score(raw)


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, theme=None, config={"displayModeBar": False})


def money(x):
    return f"{CUR}{x:,.0f}"


def describe(r):
    """Plain-English description for detected sync anomalies."""
    t = r["likely_type"]
    if t in ("Decimal / entry error", "Price mismatch"):
        return (
            f"{r['channel']} shows {CUR}{r['channel_price']:.2f} for {r['sku']}, while catalog master is "
            f"{CUR}{r['master_price']:.2f} (divergence: {abs(r['price_diff_pct']):.0%})."
        )
    if t == "Stale sync":
        return f"{r['channel']} synchronization delayed by {r['sync_delay_min']:.0f} minutes (target interval ~5m)."
    if t == "Oversell risk":
        return f"{r['channel']} inventory depleted to {r['channel_stock']:.0f} units with active velocity (stockout risk)."
    if t == "Stock drift":
        return f"{r['channel']} lists {r['channel_stock']:.0f} units, deviating from master catalog allocation."
    return f"Synchronization for {r['sku']} on {r['channel']} exhibits statistically anomalous reconstruction deviation."


# ================= SIDEBAR NAVIGATION =================
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;gap:12px;padding:4px 0 16px 0;border-bottom:1px solid {ui.BORDER};margin-bottom:12px;">'
    f'{ui.logo_html(36)}'
    f'<div>'
    f'<div style="font-size:1.05rem;font-weight:700;letter-spacing:0.04em;color:{ui.TEXT_MAIN};">SYNCGUARD <span style="color:{ui.PRIMARY};">//</span> AI</div>'
    f'<div style="font-size:0.725rem;color:{ui.TEXT_MUTED};font-family:monospace;">TELEMETRY ENGINE V2.4</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

PAGES = [
    "📊 Executive Overview",
    "🚨 Incident Triage",
    "🔭 Telemetry & Replay",
    "🧠 Neural Engine Lab",
    "⚡ API Gateway",
    "📑 Reports & Diagnostics",
]
current_page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.markdown(f'<div style="margin:16px 0 8px 0;border-top:1px solid {ui.BORDER};"></div>', unsafe_allow_html=True)

# Data source section in sidebar
st.sidebar.markdown(
    f'<div style="font-size:0.7rem;font-weight:700;color:{ui.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Sync Feed Source</div>',
    unsafe_allow_html=True,
)
upload = st.sidebar.file_uploader("Upload Sync Log", type="csv", label_visibility="collapsed")
if upload:
    raw0 = pd.read_csv(upload)
    missing = [c for c in REQUIRED if c not in raw0.columns]
    if missing:
        st.error(f"Missing required schema columns: {missing}")
        st.stop()
    raw, quality = clean(raw0)
    st.sidebar.caption(f"● Custom feed: {quality['rows_out']:,} records")
else:
    raw = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=["timestamp"])
    quality = analysis.load_json(config.QUALITY_FILE)
    st.sidebar.caption("● Active stream: 64,800 records")

det = get_detector()

# Alert sensitivity in sidebar
st.sidebar.markdown(
    f'<div style="font-size:0.7rem;font-weight:700;color:{ui.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-top:14px;margin-bottom:4px;">Detection Sensitivity</div>',
    unsafe_allow_html=True,
)
level = st.sidebar.select_slider(
    "Sensitivity",
    list(LEVELS),
    value="Balanced",
    label_visibility="collapsed",
    help="Fewer: flags high-confidence anomalies. More: heightened sensitivity with higher false discovery rate.",
)

with st.sidebar.expander("Telemetry Filters", expanded=False):
    all_channels = sorted(raw["channel"].unique())
    channels = st.multiselect("Sales channels", all_channels, default=all_channels)
    severities = st.multiselect(
        "Incident severity",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
        format_func=lambda s: URGENCY[s],
    )
    sku_text = st.text_input("SKU search", "")
    dmin, dmax = raw["timestamp"].min().date(), raw["timestamp"].max().date()
    dr = st.date_input("Time window", (dmin, dmax), min_value=dmin, max_value=dmax)
    d0, d1 = (dr[0], dr[1]) if len(dr) == 2 else (dr[0], dr[0])

with st.sidebar.expander("Engine Hyperparameters", expanded=False):
    fine = st.slider("Fine multiplier", 0.5, 2.0, 1.0, 0.05)
    use_rules = st.checkbox("Deterministic safety rules", value=config.USE_RULES)
    suppress = st.checkbox("Suppress recovery rebounds", value=config.SUPPRESS_RECOVERY)
    rule_price = st.slider("Price divergence threshold (%)", 5, 50, int(config.RULE_PRICE_DIFF * 100)) / 100
    rule_sync = st.slider("Stale delay cutoff (min)", 15, 240, config.RULE_SYNC_DELAY_MIN)
    method = st.selectbox(
        "Threshold cutoff model",
        list(det.methods),
        index=list(det.methods).index(config.THRESHOLD_METHOD),
        format_func=lambda k: thr_mod.NAMES[k],
    )

# Compute detection scores
sensitivity = LEVELS[level] * fine
base_thr = det.methods[method]
thr_eff = base_thr / sensitivity
settings = dict(
    sensitivity=sensitivity,
    use_rules=use_rules,
    rule_price=rule_price,
    rule_sync=rule_sync,
    threshold=base_thr,
    suppress_recovery=suppress,
)
res = analysis.add_context(det.apply_settings(score_data(raw), **settings))

# Filter scope
scope = res[res["timestamp"].dt.date.between(d0, d1) & res["channel"].isin(channels)]
if sku_text:
    scope = scope[scope["sku"].str.contains(sku_text, case=False)]

alerts = enrich(scope[scope["flagged"] & scope["severity"].isin(severities)]).sort_values("score", ascending=False)
statuses = store.get_all()
alerts["status"] = alerts["alert_id"].map(lambda i: statuses.get(i, ("New", ""))[0])
alerts["note"] = alerts["alert_id"].map(lambda i: statuses.get(i, ("New", ""))[1])

# Global KPI metrics across views
urgent_count = int((alerts["severity"] == "High").sum())
total_alerts = len(alerts)
unreviewed = int((alerts["status"] == "New").sum())
at_risk = alerts["est_impact"].sum()

# ================= PAGE 1: EXECUTIVE OVERVIEW =================
if current_page == "📊 Executive Overview":
    ui.hero("Executive Telemetry Command", "Real-time unsupervised sync health monitoring and risk triage across all endpoints.")

    if urgent_count > 0:
        ui.status_headline(
            f"{urgent_count:,} critical sync incidents require immediate remediation",
            f"{total_alerts:,} anomalous synchronization events detected across {len(scope):,} checked feeds.",
            badge_text=f"{urgent_count} Critical",
            badge_type="high",
        )
    else:
        ui.status_headline(
            "All synchronization pipelines are operating within nominal thresholds",
            f"Zero critical deviations detected across {len(scope):,} checked synchronization events.",
            badge_text="Operational",
            badge_type="success",
        )

    # 4 Primary KPI cards
    k = st.columns(4)
    ui.kpi(k[0], "Detected Anomalies", f"{total_alerts:,}", f"across {len(scope):,} sync events checked")
    ui.kpi(k[1], "Critical Priority", f"{urgent_count:,}", "immediate intervention required", status="danger" if urgent_count else "neutral")
    ui.kpi(k[2], "Capital Exposure", money(at_risk), "estimated financial at-risk volume", status="warning" if at_risk > 0 else "neutral")
    ui.kpi(k[3], "Unresolved Triage", f"{unreviewed:,}", "incidents awaiting confirmation")

    # Ranked "Fix These First"
    ui.section("Priority Remediation Queue", "Top 5 critical discrepancies ranked by estimated revenue exposure.")
    top5 = alerts[alerts["status"] == "New"].head(5)
    if top5.empty:
        ui.empty_state("Remediation queue clear", "No unresolved high-severity synchronization incidents.")
    else:
        tbl5 = pd.DataFrame(
            {
                "Priority": top5["severity"].map(lambda s: URGENCY[s]),
                "Product SKU": top5["sku"],
                "Channel": top5["channel"],
                "Incident Profile": top5["likely_type"],
                "Capital Exposure": top5["est_impact"],
                "Suggested Mitigation": top5["action"],
            }
        )
        st.dataframe(
            tbl5,
            hide_index=True,
            width="stretch",
            column_config={"Capital Exposure": st.column_config.NumberColumn(format=f"{CUR}%.0f")},
        )

    # ONE Primary Trend Chart
    ui.section("Telemetry Incident Trajectory", "Daily anomaly frequency categorized by severity grading.")
    if len(alerts):
        show(charts.trend(alerts), "exec_trend")
    else:
        ui.empty_state("No incident history", "Adjust date range or channel filters to inspect historical telemetry.")

    with st.expander("Expanded Diagnostic Telemetry (Channel Health & Distribution)"):
        ui.section("Endpoint Health Assessment", "Dynamic health quotient and error density per active endpoint.")
        cols = st.columns(max(len(channels), 1))
        for col, ch in zip(cols, channels):
            s_ch, a_ch = scope[scope["channel"] == ch], alerts[alerts["channel"] == ch]
            health = max(0, 100 * (1 - min(len(a_ch) / max(len(s_ch), 1) * 10, 1)))
            label, status = (
                ("Healthy", "success")
                if health >= 80
                else ("Needs Attention", "warning")
                if health >= 60
                else ("Degraded", "danger")
            )
            ui.kpi(col, ch, f"{label}", f"{len(a_ch):,} events · {money(a_ch['est_impact'].sum())} exposure", status=status)

        t1, t2, t3 = st.tabs(["By Anomaly Profile", "SKU & Channel Density", "Temporal Distribution"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Incidents grouped by error category")
                if len(alerts):
                    show(charts.type_bar(alerts), "exec_type")
            with c2:
                st.caption("Capital at risk grouped by error category")
                if len(alerts):
                    show(charts.impact_bar(alerts), "exec_impact")
            st.caption("Severity proportions")
            if len(alerts):
                show(charts.severity_donut(alerts), "exec_donut")
        with t2:
            st.caption("Top 20 products experiencing frequent feed divergence")
            if len(alerts):
                show(charts.heatmap(alerts), "exec_heat")
        with t3:
            st.caption("Incidents mapped across day-of-week and hour intervals")
            if len(alerts):
                show(charts.hour_heat(alerts), "exec_hour")


# ================= PAGE 2: INCIDENT TRIAGE =================
elif current_page == "🚨 Incident Triage":
    ui.hero("Incident Triage Workspace", "Inspect individual synchronization failures, apply corrective state, and audit resolution notes.")

    c_filter, c_actions = st.columns([2, 3])
    with c_filter:
        show_what = st.radio(
            "Triage status",
            ["To review", "Done", "All"],
            horizontal=True,
            help="'To review': New & Acknowledged. 'Done': Resolved & False positive.",
        )
    with c_actions:
        st.write("")
        b1, b2, b3 = st.columns([1.5, 1.2, 1.3])
        wanted = {"To review": ["New", "Acknowledged"], "Done": ["Resolved", "False positive"], "All": store.STATUSES}[show_what]
        ty_all = sorted(alerts["likely_type"].unique())

    with st.expander("Filter by error classification", expanded=False):
        ty_pick = st.multiselect("Classifications", ty_all, default=ty_all)

    av = alerts[alerts["status"].isin(wanted) & alerts["likely_type"].isin(ty_pick)].head(1000).reset_index(drop=True)

    with b1:
        if st.button("Mark all critical as acknowledged", disabled=av.empty, width="stretch"):
            store.set_status(av.loc[av["severity"] == "High", "alert_id"], "Acknowledged", "bulk")
            st.rerun()
    with b2:
        st.download_button(
            "⬇️ Export incidents",
            av.drop(columns=[c for c in av.columns if c.startswith("err_")]).to_csv(index=False),
            "incidents.csv",
            disabled=av.empty,
            width="stretch",
        )
    with b3:
        tech = st.toggle("ML Diagnostic columns", help="Exposes model detector source and raw reconstruction scores.")

    if av.empty:
        ui.empty_state("No incidents found", "Zero synchronization anomalies match current triage criteria.")
    else:
        tbl = pd.DataFrame(
            {
                "Severity": av["severity"].map(URGENCY),
                "Status": av["status"],
                "Timestamp": av["timestamp"],
                "Product SKU": av["sku"],
                "Channel": av["channel"],
                "Classification": av["likely_type"],
                "Capital Exposure": av["est_impact"],
            }
        )
        if tech:
            tbl["Engine"] = av["source"]
            tbl["Score"] = av["score"]

        cfg = {
            "Timestamp": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
            "Capital Exposure": st.column_config.NumberColumn(format=f"{CUR}%.0f"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
        }
        st.caption(f"Showing {len(av):,} records (sorted by anomaly score). Select an incident row to inspect details.")

        try:
            ev = st.dataframe(
                tbl,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                width="stretch",
                height=280,
                column_config=cfg,
                key="triage_table",
            )
            sel = ev.selection.rows
        except TypeError:
            st.dataframe(tbl, hide_index=True, width="stretch", height=280, column_config=cfg)
            sel = []

        row = av.loc[sel[0] if sel else 0]

        ui.section("Incident Detail Inspection", f"{row['sku']} on {row['channel']} · {row['timestamp']:%b %d, %H:%M}")

        pill_html = ui.pill(row["severity"], URGENCY[row["severity"]])
        st.markdown(
            f'<div class="sg-card">'
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
            f'{pill_html} <span style="font-weight:700;font-size:1.05rem;color:{ui.TEXT_MAIN};">{row["likely_type"]}</span>'
            f'</div>'
            f'<div style="font-size:0.925rem;line-height:1.55;color:{ui.TEXT_BODY};margin-bottom:12px;">{describe(row)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        m = st.columns(4)
        m[0].metric(
            "Channel Price",
            f"{CUR}{row['channel_price']:.2f}",
            f"{row['channel_price'] - row['master_price']:+.2f} vs master",
        )
        m[1].metric(
            "Channel Stock",
            f"{row['channel_stock']:.0f}",
            f"{row['channel_stock'] - row['master_stock'] / len(config.CHANNELS):+.0f} vs fair share",
        )
        m[2].metric("Sync Latency", f"{row['sync_delay_min']:.0f} min")
        m[3].metric("Capital Exposure", money(row["est_impact"]))

        ui.tip(f"**Recommended Mitigation:** {row['action']}")

        s1, s2, s3 = st.columns([2, 3, 1])
        new_status = s1.selectbox("Remediation Status", store.STATUSES, index=store.STATUSES.index(row["status"]))
        note = s2.text_input("Audit Note", row["note"])
        s3.write("")
        s3.write("")
        if s3.button("Update Status", width="stretch"):
            store.set_status([row["alert_id"]], new_status, note)
            st.rerun()

        with st.expander("Reconstruction Error Attribution (Model Diagnostics)"):
            st.caption("Breakdown of dimension-level reconstruction divergence and threshold gauge.")
            a, b, c = st.columns([1.2, 1, 1])
            with a:
                st.markdown("**Feature Deviation vs Bound**")
                show(charts.contribution(row), "tr_contrib")
            with b:
                st.markdown("**Anomaly Score Index**")
                show(charts.gauge(row["score"], thr_eff), "tr_gauge")
            with c:
                st.markdown("**Deviation Radar Profile**")
                show(charts.radar(row), "tr_radar")
            st.caption(f"Flagged by Engine: {row['source']} (Model = Autoencoder, Rule = Safety rule, Both = Ensemble)")

        with st.expander("Ops Notification Payload"):
            st.code(
                f"INCIDENT [{URGENCY[row['severity']]}] // {row['likely_type']} on {row['channel']}\n"
                f"TARGET SKU: {row['sku']} at {row['timestamp']:%Y-%m-%d %H:%M}\n"
                f"DIAGNOSTIC: {describe(row)}\n"
                f"CAPITAL AT RISK: {money(row['est_impact'])}\n"
                f"RECOMMENDED MITIGATION: {row['action']}",
                language=None,
            )


# ================= PAGE 3: TELEMETRY & REPLAY =================
elif current_page == "🔭 Telemetry & Replay":
    ui.hero("Diagnostic Telemetry & Simulation", "Analyze historical SKU curves, simulate real-time live replay streams, and test scenario hypotheses.")

    tool = st.radio(
        "Telemetry Mode",
        ["Product history", "Live stream replay", "Scenario simulation"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if tool == "Product history":
        ui.section("SKU Trajectory Analysis", "Examine historical price, inventory, and sync delay dynamics for a designated SKU.")
        x1, x2 = st.columns(2)
        sku_pick = x1.selectbox("Select Target SKU", sorted(res["sku"].unique()))
        ch_pick = x2.selectbox("Target Sales Channel", sorted(res["channel"].unique()))
        one = res[res["sku"] == sku_pick].sort_values("timestamp")
        d = one[one["channel"] == ch_pick]
        sa = enrich(one[one["flagged"]])

        k = st.columns(3)
        ui.kpi(k[0], "SKU Anomaly Count", f"{len(sa):,}", "across all sales endpoints")
        ui.kpi(k[1], "Cumulative Exposure", money(sa["est_impact"].sum()), "accumulated capital exposure")
        ui.kpi(k[2], "Mean Sync Latency", f"{d['sync_delay_min'].mean():.1f} min", "benchmark is ~5.0m")

        st.markdown("**Price Curve** (solid = channel feed, dashed = master catalog, markers = flagged anomalies)")
        show(charts.timeline(d, "channel_price", "Channel Price", "master_price"), "ph_price")

        with st.expander("Secondary Dimension Telemetry (Inventory, Sync Latency, Cross-Channel)"):
            st.markdown("**Inventory Curve**")
            show(charts.timeline(d, "channel_stock", "Channel Stock"), "ph_stock")
            st.markdown("**Sync Latency (minutes)**")
            show(charts.timeline(d, "sync_delay_min", "Sync Latency (min)"), "ph_sync")
            st.markdown("**Cross-Channel Comparative Pricing**")
            show(charts.cross_channel(one), "ph_cross")

        if len(sa):
            with st.expander(f"Incident log for {sku_pick}"):
                st.dataframe(
                    sa[["timestamp", "channel", "severity", "likely_type", "est_impact"]].sort_values("timestamp", ascending=False),
                    hide_index=True,
                    width="stretch",
                    height=220,
                )

    elif tool == "Live stream replay":
        ui.section("Real-Time Stream Playback", "Simulate live streaming feed playback to audit detection throughput and latency.")
        l1, l2, l3 = st.columns([1, 1, 1])
        window = l1.slider("Replay window (hours)", 12, 168, 48)
        delay = l2.slider("Interval delay (sec/hour)", 0.05, 1.0, 0.15)
        l3.write("")
        l3.write("")
        if l3.button("Execute Playback", type="primary", width="stretch"):
            hours = np.sort(scope["timestamp"].unique())[-window:]
            ph_k, ph_c, ph_f, bar = st.empty(), st.empty(), st.empty(), st.progress(0)
            series, feed, total, impact = [], [], 0, 0.0
            for i, t in enumerate(hours):
                e = enrich(scope[(scope["timestamp"] == t) & scope["flagged"]])
                total += len(e)
                impact += e["est_impact"].sum()
                series.append({"t": pd.Timestamp(t), "alerts": len(e)})
                if len(e):
                    feed.append(e)
                with ph_k.container():
                    kk = st.columns(4)
                    ui.kpi(kk[0], "Stream Cursor", f"{pd.Timestamp(t):%b %d %H:%M}", f"hour {i + 1} of {len(hours)}")
                    ui.kpi(kk[1], "Cumulative Alerts", f"{total}", "in replay window")
                    ui.kpi(kk[2], "Interval Rate", f"{len(e)}", "new anomalies", status="danger" if len(e) else "success")
                    ui.kpi(kk[3], "Accumulated Exposure", money(impact), "total risk")
                ph_c.plotly_chart(
                    charts.live_chart(pd.DataFrame(series)),
                    width="stretch",
                    key=f"live_{i}",
                    theme=None,
                    config={"displayModeBar": False},
                )
                if feed:
                    f = pd.concat(feed).sort_values("timestamp", ascending=False).head(6)
                    ph_f.dataframe(
                        pd.DataFrame(
                            {
                                "Severity": f["severity"].map(URGENCY),
                                "Timestamp": f["timestamp"],
                                "SKU": f["sku"],
                                "Channel": f["channel"],
                                "Classification": f["likely_type"],
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                bar.progress((i + 1) / len(hours))
                time.sleep(delay)
            st.success("Replay sequence completed.")

    else:
        ui.section("Synthetic Stress Testing", "Simulate custom sync anomalies and verify neural detection sensitivity.")
        PRESETS = {
            "Nominal Sync": (0, 0, 0, 5),
            "50% Price Collapse": (-50, 0, 0, 5),
            "Decimal Typo (÷10)": (-90, 0, 0, 5),
            "Depleted Stock + Velocity": (0, -99, 20, 5),
            "Sync Stalled 4h": (0, 0, 0, 240),
            "Inventory Count Drift": (0, -60, 0, 5),
        }
        KEYS = ["wi_price", "wi_stock", "wi_units", "wi_sync"]
        for k_, v_ in zip(KEYS, PRESETS["Nominal Sync"]):
            st.session_state.setdefault(k_, v_)

        def set_preset(name):
            for k_, v_ in zip(KEYS, PRESETS[name]):
                st.session_state[k_] = v_

        w1, w2 = st.columns(2)
        wsku = w1.selectbox("Target SKU", sorted(raw["sku"].unique()), key="wi_sku")
        wch = w2.selectbox("Target Channel", sorted(raw["channel"].unique()), key="wi_ch")

        st.caption("Quick Preset Failure Modes")
        pb = st.columns(len(PRESETS))
        for col, name in zip(pb, PRESETS):
            col.button(name, on_click=set_preset, args=(name,), width="stretch")

        with st.expander("Custom Stress Vector Parameters", expanded=True):
            s1, s2, s3, s4 = st.columns(4)
            pp = s1.slider("Price shift (%)", -95, 100, key="wi_price")
            sp = s2.slider("Stock shift (%)", -100, 200, key="wi_stock")
            eu = s3.slider("Velocity spike (units/hr)", 0, 40, key="wi_units")
            sy = s4.slider("Sync latency (minutes)", 0, 600, key="wi_sync")

        base = raw[(raw["sku"] == wsku) & (raw["channel"] == wch)].sort_values("timestamp").tail(24)
        sim = base.iloc[[-1]].copy()
        sim["timestamp"] = sim["timestamp"] + pd.Timedelta(hours=1)
        sim["channel_price"] = round(float(sim["channel_price"].iloc[0]) * (1 + pp / 100), 2)
        sim["channel_stock"] = max(round(float(sim["channel_stock"].iloc[0]) * (1 + sp / 100)), 0)
        sim["units_sold"] = sim["units_sold"] + eu
        sim["sync_delay_min"] = sy
        out = enrich(det.apply_settings(det.score(pd.concat([base, sim])), **settings).tail(1)).iloc[0]

        ui.section("Inference Verdict", "Neural autoencoder classification and recommended mitigation.")
        if out["flagged"]:
            ui.status_headline(
                f"Anomaly Flagged: {out['likely_type']}",
                describe(out),
                badge_text=f"{URGENCY[out['severity']]} Priority",
                badge_type="high" if out["severity"] == "High" else "medium",
            )
            ui.tip(f"**Action Recommended:** {out['action']}")
        else:
            ui.status_headline(
                "Telemetry Nominal",
                "Reconstruction error remained strictly within learned baseline bounds.",
                badge_text="Nominal",
                badge_type="success",
            )

        with st.expander("Score Diagnostics & Parameter Comparison"):
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Anomaly Score vs Cutoff Threshold")
                show(charts.gauge(out["score"], thr_eff), "wi_gauge")
            with g2:
                st.caption("Baseline Feed vs Simulated Vector")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Parameter": ["Channel Price", "Channel Stock", "Sales Velocity", "Sync Latency"],
                            "Baseline": [
                                f"{CUR}{base['channel_price'].iloc[-1]:.2f}",
                                f"{base['channel_stock'].iloc[-1]:.0f}",
                                f"{base['units_sold'].iloc[-1]}",
                                f"{base['sync_delay_min'].iloc[-1]:.1f}m",
                            ],
                            "Simulated": [
                                f"{CUR}{out['channel_price']:.2f}",
                                f"{out['channel_stock']:.0f}",
                                f"{out['units_sold']}",
                                f"{sy:.1f}m",
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )


# ================= PAGE 4: NEURAL ENGINE LAB =================
elif current_page == "🧠 Neural Engine Lab":
    ui.hero("Neural Autoencoder Laboratory", "Inspect dataset hygiene, unsupervised 6→4→6 compression topology, threshold optimization, and held-out validation.")

    has_labels = "is_anomaly" in res.columns
    hold = (res if upload else holdout(res)) if has_labels else None
    lab.render(
        {
            "clean": raw,
            "quality": quality,
            "res": res,
            "hold": hold,
            "det": det,
            "thr": thr_eff,
            "upload": upload is not None,
            "art": analysis.artifacts(),
            "scored": score_data(raw),
            "settings": settings,
        }
    )


# ================= PAGE 5: API GATEWAY =================
elif current_page == "⚡ API Gateway":
    ui.hero("Scoring Microservice Gateway", "Deploy low-latency synchronization anomaly inference to ERP connectors, catalog schedulers, and edge proxies.")
    ui.tip("Start the FastAPI server via `uvicorn api:app --reload` and explore docs at **http://127.0.0.1:8000/docs**.")

    st.dataframe(
        pd.DataFrame(
            [
                ("GET", "/health", "Service uptime and autoencoder memory footprint"),
                ("GET", "/model/info", "Network topology, weights baseline, and current threshold"),
                ("GET", "/thresholds", "Threshold method benchmark comparison"),
                ("GET", "/metrics", "Held-out test set precision, recall, and F1 metrics"),
                ("POST", "/score", "Synchronously score a single feed record"),
                ("POST", "/score/batch", "Batch score up to 5,000 synchronization events"),
            ],
            columns=["Method", "Route", "Telemetry Service Description"],
        ),
        hide_index=True,
        width="stretch",
    )

    ui.section("Live Endpoint Tester", "Dispatch test payloads directly against the scoring module.")
    base = (raw[raw["is_anomaly"] == 0] if "is_anomaly" in raw.columns else raw).iloc[500]
    payload0 = {"timestamp": base["timestamp"].isoformat(), "sku": base["sku"], "channel": base["channel"]}
    payload0.update(
        {
            k_: float(base[k_])
            for k_ in ["master_price", "channel_price", "master_stock", "channel_stock", "units_sold", "sync_delay_min"]
        }
    )
    SAMPLES = {
        "Nominal Feed Sync": {},
        "Decimal Typo (÷10)": {"channel_price": round(payload0["channel_price"] / 10, 2)},
        "Stale Feed (5h Delayed)": {"sync_delay_min": 300.0},
        "Depleted Stock + Velocity Spike": {"channel_stock": 1.0, "units_sold": payload0["units_sold"] + 20},
    }
    pick = st.radio("Simulation Payload", list(SAMPLES), horizontal=True)
    payload = {**payload0, **SAMPLES[pick]}

    a, b = st.columns(2)
    with a:
        st.markdown("**Request Payload** · `POST /score`")
        st.code(json.dumps(payload, indent=2), language="json")
    with b:
        out = service.score_records([payload])[0]
        st.markdown("**Gateway Response**")
        if out["is_anomaly"]:
            st.error(f"⚡ ANOMALY CONFIRMED · Score {out['anomaly_score']} (cutoff {out['threshold']})")
        else:
            st.success(f"✓ NOMINAL SYNCHRONIZATION · Score {out['anomaly_score']} (cutoff {out['threshold']})")
        st.json(out, expanded=False)

    with st.expander("Client Integration Snippets (Python, cURL, Container)"):
        st.markdown("**Python requests**")
        st.code(
            "import requests\n"
            "res = requests.post('http://127.0.0.1:8000/score', json=payload)\n"
            "print(res.json())",
            language="python",
        )
        st.markdown("**cURL**")
        st.code(
            "curl -X POST http://127.0.0.1:8000/score \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d @payload.json",
            language="bash",
        )
        st.markdown("**Containerized Deployment**")
        st.code("docker build -t syncguard .\ndocker run -p 8000:8000 syncguard", language="bash")


# ================= PAGE 6: REPORTS & DIAGNOSTICS =================
elif current_page == "📑 Reports & Diagnostics":
    ui.hero("Deliverables & Technical Documentation", "Download audited analytical reports, review system architecture, and inspect modeling assumptions.")

    with st.expander("Exportable Deliverables & Dataset Downloads", expanded=True):
        for fname, label in [
            ("01_data_preparation_report.html", "Dataset Preparation Report (Milestone 1)"),
            ("02_final_project_report.html", "Final Project Report (Milestone 4)"),
        ]:
            pth = os.path.join(config.REPORTS_DIR, fname)
            if os.path.exists(pth):
                st.download_button(
                    f"⬇️ {label}",
                    open(pth, encoding="utf-8").read(),
                    fname,
                    mime="text/html",
                    width="stretch",
                )

        r1, r2 = st.columns(2)
        top = alerts.head(10)[["timestamp", "sku", "channel", "severity", "likely_type", "est_impact"]]
        by_type = alerts.groupby("likely_type").agg(alerts=("score", "size"), money_at_risk=("est_impact", "sum"))
        report_html = (
            f"<html><body style='font-family:Arial;max-width:900px;margin:30px auto;background:#111;color:#eee;'>"
            f"<h1>SyncGuard AI: Diagnostic Summary</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M}</p>"
            f"<ul><li>Syncs checked: {len(scope):,}</li><li>Discrepancies found: {len(alerts):,}</li>"
            f"<li>Critical priority: {urgent_count:,}</li>"
            f"<li>Estimated revenue exposure: {money(at_risk)}</li></ul>"
            f"<h2>By Discrepancy Type</h2>{by_type.round(0).to_html()}<h2>Top 10 Priority Issues</h2>{top.round(2).to_html(index=False)}"
            f"</body></html>"
        )
        r1.download_button("⬇️ Executive Summary (HTML)", report_html, "syncguard_summary.html", mime="text/html", width="stretch")
        r2.download_button(
            "⬇️ Full Telemetry Log (CSV)",
            alerts.drop(columns=[c for c in alerts.columns if c.startswith("err_")]).to_csv(index=False),
            "alerts_full.csv",
            width="stretch",
        )

    with st.expander("Mission Control Architecture"):
        st.graphviz_chart(
            """digraph { rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#192033", color="#00F2FE", fontcolor="#F0F4FC", fontname="Inter"];
            edge [color="#687794"];
            "Multi-Channel Sync Streams" -> "6 Scaled Features" -> "Autoencoder (6→4→6)" -> "Reconstruction Divergence" -> "Dynamic Cutoff + Rules" -> "Incident Triage";
            "Incident Triage" -> "Remediation State (SQLite)"; }"""
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature Dimension": [charts.LABELS[c] for c in config.FEATURE_COLUMNS],
                    "Targeted Failure Mode": [config.REASONS[c] for c in config.FEATURE_COLUMNS],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Telemetry Terminology Reference"):
        st.markdown("""
- **Feed Event**: Periodic sync communication between enterprise master catalog and sales endpoint (Amazon, Shopify, eBay).
- **Master Catalog**: The authoritative, centralized enterprise pricing and stock record.
- **Severity Index**: Quantitative divergence magnitude mapped to commercial revenue risk.
- **Stale Feed**: Channel synchronization that has failed to update beyond SLA latency (>60 minutes).
- **Oversell Risk**: Channel inventory critically low while sales velocity continues, risking stockouts and cancellations.
- **Autoencoder Reconstruction Error**: Mathematical measure of input deviation from learned normal patterns; high divergence indicates an anomaly.
""")

    with st.expander("Database Maintenance"):
        st.caption("Purge stored triage resolution audit records.")
        if st.button("Reset Triage States"):
            store.reset()
            st.rerun()
