"""Plotly charts for SyncGuard AI: Cyber Analytics / Dark Mode.
Dark transparent background, glowing neon traces, subtle gridlines, and high-contrast telemetry.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config

# Cyber Design System Palette
PRIMARY = "#00F2FE"          # Electric Cyan
PRIMARY_TINT = "rgba(0, 242, 254, 0.12)"
NEUTRAL_GRID = "rgba(255, 255, 255, 0.07)"
NEUTRAL_LINE = "rgba(255, 255, 255, 0.12)"
NEUTRAL_TEXT = "#687794"      # Muted tick text
TEXT_MAIN = "#F0F4FC"
BAR_MUTED = "#222B3D"         # Normal/unselected dark bar

# Semantic Laser Palette
SUCCESS = "#00E599"          # Electric Emerald
WARNING = "#FFB800"          # Neon Amber
DANGER = "#FF3B5C"           # Laser Red
CYBER_VIOLET = "#8A2BE2"     # Neon Purple

SEV = {
    "High": DANGER,
    "Medium": WARNING,
    "Low": PRIMARY,
}

# Channels
CHANNEL_COLORS = [PRIMARY, SUCCESS, CYBER_VIOLET]

LABELS = {
    "price_diff_pct": "Price gap",
    "stock_diff_pct": "Stock gap",
    "sync_delay_log": "Sync delay",
    "stock_cover_hours": "Stock cover",
    "price_change_pct": "Price jump",
    "stock_change_pct": "Stock jump",
    "sync_attempts_log": "Sync retries",
    "update_gap_log": "Update interval",
    "sync_status_code": "Sync status",
}


def _style(fig, h=280, legend=True):
    """Shared Dark Cyber Plotly styling template."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=h,
        margin=dict(l=14, r=14, t=20, b=20),
        showlegend=legend,
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#A2B0C7"),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0,
            xanchor="left",
            font=dict(size=11, color=NEUTRAL_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=NEUTRAL_LINE,
        linewidth=1,
        tickfont=dict(size=11, color=NEUTRAL_TEXT),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=NEUTRAL_GRID,
        gridwidth=1,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color=NEUTRAL_TEXT),
        zeroline=False,
    )
    return fig


def trend(alerts):
    """Daily alerts bar chart by severity."""
    d = (
        alerts.assign(day=alerts["timestamp"].dt.floor("D"))
        .groupby(["day", "severity"])
        .size()
        .reset_index(name="alerts")
    )
    fig = px.bar(
        d,
        x="day",
        y="alerts",
        color="severity",
        color_discrete_map=SEV,
        category_orders={"severity": ["High", "Medium", "Low"]},
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Alerts / day",
        bargap=0.25,
    )
    return _style(fig, 260)


def severity_donut(alerts):
    """Donut chart showing severity breakdown."""
    c = alerts["severity"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    fig = go.Figure(
        go.Pie(
            labels=c.index,
            values=c.values,
            hole=0.68,
            sort=False,
            marker=dict(colors=[SEV[k] for k in c.index], line=dict(color="#0A0D14", width=2)),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=11, color="#A2B0C7"),
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return _style(fig, 250, legend=False)


def type_bar(alerts):
    """Horizontal bar chart of issue frequency by type."""
    c = alerts["likely_type"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=c.values,
            y=c.index,
            orientation="h",
            marker=dict(color=PRIMARY, line=dict(color="rgba(0, 242, 254, 0.6)", width=1)),
            text=[f"{v:,}" for v in c.values],
            textposition="auto",
            textfont=dict(color="#0A0D14", size=10),
        )
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return _style(fig, 240, False)


def impact_bar(alerts):
    """Horizontal bar chart of estimated revenue impact by issue type."""
    c = alerts.groupby("likely_type")["est_impact"].sum().sort_values()
    fig = go.Figure(
        go.Bar(
            x=c.values,
            y=c.index,
            orientation="h",
            marker=dict(color=WARNING, line=dict(color="rgba(255, 184, 0, 0.6)", width=1)),
            text=[f"{config.CURRENCY}{v:,.0f}" for v in c.values],
            textposition="auto",
            textfont=dict(color="#0A0D14", size=10),
        )
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return _style(fig, 240, False)


def heatmap(alerts):
    """Heatmap showing issues across top 20 SKUs and channels."""
    top = alerts["sku"].value_counts().head(20).index
    p = alerts[alerts["sku"].isin(top)].pivot_table(
        index="sku", columns="channel", values="score", aggfunc="count", fill_value=0
    )
    fig = px.imshow(
        p,
        aspect="auto",
        color_continuous_scale=[[0, "#111726"], [0.5, "#1B2A4A"], [1, PRIMARY]],
        text_auto=True,
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
    return _style(fig, 360, False)


def hour_heat(alerts):
    """Heatmap of alerts by weekday and hour."""
    d = alerts.assign(hour=alerts["timestamp"].dt.hour, dow=alerts["timestamp"].dt.day_name())
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    p = (
        d.pivot_table(index="dow", columns="hour", values="score", aggfunc="count", fill_value=0)
        .reindex(order)
        .fillna(0)
    )
    fig = px.imshow(
        p,
        aspect="auto",
        color_continuous_scale=[[0, "#111726"], [0.5, "#1B2A4A"], [1, PRIMARY]],
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Hour of day", yaxis_title=None)
    return _style(fig, 260, False)


def contribution(row):
    """Bar chart indicating which feature exceeded its normal bound."""
    vals = pd.Series({LABELS[c]: row["err_" + c] for c in config.FEATURE_COLUMNS}).sort_values()
    fig = go.Figure(
        go.Bar(
            x=vals.values,
            y=vals.index,
            orientation="h",
            marker_color=[DANGER if v > 1 else BAR_MUTED for v in vals.values],
            marker_line=dict(color=[DANGER if v > 1 else "rgba(255,255,255,0.1)" for v in vals.values], width=1),
        )
    )
    fig.add_vline(x=1, line_dash="dot", line_color=PRIMARY, annotation_text="Limit")
    fig.update_layout(xaxis_title="Deviation vs normal bound", yaxis_title=None)
    return _style(fig, 240, False)


def radar(row):
    """Radar chart showing feature anomaly profile."""
    names = [LABELS[c] for c in config.FEATURE_COLUMNS]
    vals = [float(min(row["err_" + c], 6)) for c in config.FEATURE_COLUMNS]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[1] * 7,
            theta=names + names[:1],
            name="Normal limit",
            line=dict(color=PRIMARY, dash="dot", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=vals + vals[:1],
            theta=names + names[:1],
            fill="toself",
            fillcolor="rgba(255, 59, 92, 0.2)",
            name="Deviation",
            line=dict(color=DANGER, width=2),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, max(max(vals) * 1.1, 2)], linecolor=NEUTRAL_LINE, gridcolor=NEUTRAL_GRID),
            angularaxis=dict(linecolor=NEUTRAL_LINE, gridcolor=NEUTRAL_GRID),
        )
    )
    return _style(fig, 260)


def timeline(d, col, title, master_col=None):
    """Timeline line chart for product metrics."""
    fig = go.Figure(
        go.Scatter(
            x=d["timestamp"],
            y=d[col],
            mode="lines",
            name=title,
            line=dict(color=PRIMARY, width=2),
        )
    )
    if master_col:
        fig.add_trace(
            go.Scatter(
                x=d["timestamp"],
                y=d[master_col],
                mode="lines",
                name="Master",
                line=dict(color=NEUTRAL_TEXT, dash="dot", width=1.2),
            )
        )
    f = d[d["flagged"]]
    fig.add_trace(
        go.Scatter(
            x=f["timestamp"],
            y=f[col],
            mode="markers",
            name="Flagged",
            marker=dict(color=DANGER, size=8, line=dict(color="#FFFFFF", width=1.5)),
        )
    )
    fig.update_layout(yaxis_title=title, xaxis_title=None)
    return _style(fig, 240)


def cross_channel(d):
    """Multi-line comparison across channels."""
    fig = px.line(
        d,
        x="timestamp",
        y="channel_price",
        color="channel",
        color_discrete_sequence=CHANNEL_COLORS,
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Price")
    return _style(fig, 240)


def gauge(score, thr):
    """Score indicator gauge."""
    top = max(thr * 3.5, score * 1.15)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number=dict(valueformat=".2f", font=dict(family="JetBrains Mono", size=24, color=TEXT_MAIN)),
            gauge=dict(
                axis=dict(range=[0, top], tickcolor=NEUTRAL_TEXT),
                bar=dict(color=PRIMARY, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0, thr], color="rgba(0, 229, 153, 0.15)"),
                    dict(range=[thr, thr * config.SEVERITY_MEDIUM], color="rgba(255, 184, 0, 0.15)"),
                    dict(range=[thr * config.SEVERITY_MEDIUM, top], color="rgba(255, 59, 92, 0.2)"),
                ],
                threshold=dict(line=dict(color=DANGER, width=3), value=thr),
            ),
        )
    )
    return _style(fig, 220, False)


def confusion(y, p):
    """Confusion matrix heatmap."""
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(
        go.Heatmap(
            z=np.log10(np.array(z) + 1),
            x=["Pred Normal", "Pred Alert"],
            y=["Actual Normal", "Actual Error"],
            text=[[f"{v:,}" for v in r] for r in z],
            texttemplate="%{text}",
            textfont=dict(size=15, family="JetBrains Mono", color=TEXT_MAIN),
            colorscale=[[0, "#111726"], [0.5, "#1B2A4A"], [1, PRIMARY]],
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 260, False)


def score_hist(h, thr):
    """Log-scale anomaly score distribution."""
    h = h.assign(ls=np.log10(h["score"].clip(lower=1e-3)))
    fig = px.histogram(
        h,
        x="ls",
        color=h["is_anomaly"].map({0: "Normal", 1: "Real error"}),
        nbins=70,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"Normal": BAR_MUTED, "Real error": DANGER},
    )
    fig.add_vline(x=np.log10(thr), line_color=PRIMARY, line_width=2, annotation_text="Threshold")
    fig.update_layout(xaxis_title="Score (log scale)", yaxis_title="Rows", yaxis_type="log")
    return _style(fig, 260)


def sweep(df):
    """Sensitivity evaluation curves."""
    fig = go.Figure()
    curves = [("Precision", SUCCESS), ("Recall", PRIMARY), ("F1", WARNING)]
    for k, c in curves:
        fig.add_trace(
            go.Scatter(x=df["sensitivity"], y=df[k], mode="lines+markers", name=k, line=dict(color=c, width=2))
        )
    fig.update_layout(xaxis_title="Sensitivity", yaxis_title="Metric", yaxis_range=[0, 1.02])
    return _style(fig, 260)


def type_recall(df):
    """Recall rate by anomaly type."""
    fig = go.Figure(
        go.Bar(
            x=df.index,
            y=df.values,
            marker_color=PRIMARY,
            text=[f"{v:.0%}" for v in df.values],
            textposition="outside",
            textfont=dict(color=TEXT_MAIN),
        )
    )
    fig.update_layout(yaxis_range=[0, 1.15], yaxis_tickformat=".0%", xaxis_title=None)
    return _style(fig, 260, False)


def type_matrix(m):
    """Classification type mapping matrix."""
    fig = px.imshow(
        m,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, "#111726"], [0.5, "#1B2A4A"], [1, PRIMARY]],
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Predicted", yaxis_title="True")
    return _style(fig, 280, False)


def live_chart(df):
    """Live stream line chart."""
    fig = go.Figure(
        go.Scatter(
            x=df["t"],
            y=df["alerts"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor=PRIMARY_TINT,
            line=dict(color=PRIMARY, width=2),
            marker=dict(size=4),
        )
    )
    fig.update_layout(yaxis_title="Alerts / hr", xaxis_title=None)
    return _style(fig, 220, False)


# ---------- Lab charts ----------
SHORT = {
    "percentile": "Percentile",
    "mean_3sd": "Mean+3σ",
    "mad": "Median+3·MAD",
    "iqr": "Q3+3·IQR",
    "f1_optimal": "Best-F1 (ref.)",
}


def price_ratio_box(d):
    """Price ratio boxplot by sales channel."""
    d = d.assign(ratio=d["channel_price"] / d["master_price"])
    fig = px.box(
        d.sample(min(len(d), 20000), random_state=1),
        x="channel",
        y="ratio",
        color="channel",
        color_discrete_sequence=CHANNEL_COLORS,
    )
    fig.update_layout(yaxis_title="Channel ÷ Master", xaxis_title=None, yaxis_range=[0.85, 1.15])
    return _style(fig, 260, False)


def delay_hist(d):
    """Sync delay frequency distribution."""
    fig = px.histogram(d, x="sync_delay_min", nbins=50, range_x=[0, 60], color_discrete_sequence=[PRIMARY])
    fig.update_layout(xaxis_title="Minutes since last sync", yaxis_title="Records")
    return _style(fig, 260, False)


def stock_scatter(d):
    """Fair share vs actual channel stock scatter."""
    s = d.sample(min(len(d), 3000), random_state=1).assign(fair=lambda x: x["master_stock"] / len(config.CHANNELS))
    fig = px.scatter(
        s,
        x="fair",
        y="channel_stock",
        color="channel",
        opacity=0.55,
        color_discrete_sequence=CHANNEL_COLORS,
    )
    top = float(s["fair"].max())
    fig.add_trace(
        go.Scatter(
            x=[0, top],
            y=[0, top],
            mode="lines",
            name="Ideal sync",
            line=dict(color=TEXT_MAIN, dash="dot", width=1.5),
        )
    )
    fig.update_layout(xaxis_title="Fair share of master stock", yaxis_title="Channel stock")
    return _style(fig, 280)


def units_hist(d):
    """Hourly sales units histogram."""
    fig = px.histogram(d, x="units_sold", nbins=25, color_discrete_sequence=[WARNING], range_x=[0, 30])
    fig.update_layout(xaxis_title="Units sold / hour", yaxis_title="Records")
    return _style(fig, 260, False)


def feature_dist(f):
    """Scaled distribution comparison across all 6 features."""
    cols = config.FEATURE_COLUMNS
    m = f.melt(id_vars="is_anomaly", value_vars=cols, var_name="feature", value_name="value")
    m["feature"] = m["feature"].map(LABELS)
    m["value"] = m["value"].clip(-6, 6)
    m["class"] = m["is_anomaly"].map({0: "Normal", 1: "Real error"})
    fig = px.box(
        m.sample(min(len(m), 20000), random_state=1),
        x="class",
        y="value",
        color="class",
        facet_col="feature",
        facet_col_wrap=3,
        color_discrete_map={"Normal": BAR_MUTED, "Real error": DANGER},
    )
    fig.update_layout(yaxis_title="Scaled value")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title=None)
    return _style(fig, 420, False)


def corr_heat(f):
    """Feature correlation matrix heatmap."""
    c = f[config.FEATURE_COLUMNS].corr().round(2)
    c.index = c.columns = [LABELS[x] for x in c.columns]
    fig = px.imshow(
        c,
        text_auto=True,
        color_continuous_scale=[[0, "#111726"], [0.5, "#1B2A4A"], [1, PRIMARY]],
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, 320, False)


def loss_curve(model):
    """Training reconstruction loss curve."""
    y = model.loss_curve_
    fig = go.Figure(
        go.Scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines", line=dict(color=PRIMARY, width=2.5))
    )
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Reconstruction Loss", yaxis_type="log")
    return _style(fig, 250, False)


def arch_bars(e):
    """Architecture comparison bar chart."""
    fig = go.Figure(
        go.Bar(
            x=e["architecture"],
            y=e["f1"],
            text=[f"{v:.2f}" for v in e["f1"]],
            textposition="outside",
            marker_color=[PRIMARY if c else BAR_MUTED for c in e["chosen"]],
            textfont=dict(color=TEXT_MAIN),
        )
    )
    fig.update_layout(yaxis_title="Held-out F1", yaxis_range=[0, 1], xaxis_title=None)
    return _style(fig, 260, False)


def reconstruction(actual, recon, title):
    """Actual vs reconstructed feature comparison."""
    names = [LABELS[c] for c in config.FEATURE_COLUMNS]
    fig = go.Figure(
        [
            go.Bar(name="Actual input", x=names, y=actual, marker_color=NEUTRAL_TEXT),
            go.Bar(name="Reconstructed", x=names, y=recon, marker_color=PRIMARY),
        ]
    )
    fig.update_layout(barmode="group", yaxis_title="Scaled value")
    return _style(fig, 250)


def thr_hist(h, methods):
    """Score histogram with threshold candidate cutoffs."""
    h = h.assign(ls=np.log10(h["score"].clip(lower=1e-3)))
    fig = px.histogram(
        h,
        x="ls",
        color=h["is_anomaly"].map({0: "Normal", 1: "Real error"}),
        nbins=70,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Normal": BAR_MUTED, "Real error": DANGER},
    )
    line_palette = [PRIMARY, SUCCESS, WARNING, TEXT_MAIN, CYBER_VIOLET]
    for (k, v), c in zip(methods.items(), line_palette):
        val = float(np.log10(max(v, 1e-3)))
        fig.add_vline(x=val, line_color=c, line_width=1.8, line_dash="dash")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=SHORT.get(k, k), line=dict(color=c, dash="dash")))
    fig.update_layout(xaxis_title="Score (log scale)", yaxis_title="Rows", yaxis_type="log")
    return _style(fig, 280)


def pr_curve(y, scores, thr):
    """Precision-Recall trade-off curve."""
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y, scores)
    fig = go.Figure(go.Scatter(x=r, y=p, mode="lines", line=dict(color=PRIMARY, width=2.5), name="Autoencoder"))
    pred = np.asarray(scores) > thr
    yy = np.asarray(y)
    cp = (pred & (yy == 1)).sum() / max(pred.sum(), 1)
    cr = (pred & (yy == 1)).sum() / max((yy == 1).sum(), 1)
    fig.add_trace(
        go.Scatter(
            x=[cr],
            y=[cp],
            mode="markers",
            name="Operating point",
            marker=dict(size=11, color=DANGER, line=dict(color="#FFFFFF", width=2)),
        )
    )
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1.02], yaxis_range=[0, 1.02])
    return _style(fig, 280)


def method_bars(ev):
    """Performance metrics grouped by threshold method."""
    fig = go.Figure()
    metrics = [("precision", SUCCESS), ("recall", PRIMARY), ("f1", WARNING)]
    for k, c in metrics:
        fig.add_trace(
            go.Bar(
                x=[SHORT.get(m, m) for m in ev["method"]],
                y=ev[k],
                name=k.capitalize() if k != "f1" else "F1",
                marker_color=c,
            )
        )
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
    return _style(fig, 280)


def approach_bars(t):
    """Rules only vs Autoencoder vs Combined approach."""
    fig = go.Figure()
    colors = [BAR_MUTED, PRIMARY, SUCCESS]
    for col, c in zip(t.columns, colors):
        fig.add_trace(go.Bar(x=t.index, y=t[col], name=col, marker_color=c))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05], yaxis_tickformat=".0%")
    return _style(fig, 280)


def group_heat(g):
    """Error feature pattern matrix."""
    g = g.copy()
    g.columns = [LABELS[c] for c in g.columns]
    fig = px.imshow(
        g.round(2),
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, "#111726"], [0.5, "#1B2A4A"], [1, PRIMARY]],
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, 260, False)


def reasons_bar(s):
    """Frequency bar chart for false alarm drivers."""
    s = s.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=PRIMARY))
    return _style(fig, 230, False)


def spread_bars(cc):
    """Alert and real error rates across cross-channel spread bands."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cc.index.astype(str), y=cc["alerts"], name="Alert rate", marker_color="#F26A4B"))
    if "real_errors" in cc.columns:
        fig.add_trace(go.Bar(x=cc.index.astype(str), y=cc["real_errors"], name="Real error rate", marker_color="#8E7CC3"))
    fig.update_layout(title="Alert & Error Rate by Cross-Channel Disagreement", barmode="group",
                      xaxis_title="Cross-Channel Disagreement", yaxis_title="Rate", yaxis_tickformat=".1%")
    return _style(fig, 280)


def history_lines(hist):
    """Held-out precision, recall, and F1 across training history volumes."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["days_of_history"], y=hist["precision"], name="Precision", mode="lines+markers", line=dict(color="#2B7CE9", width=2)))
    fig.add_trace(go.Scatter(x=hist["days_of_history"], y=hist["recall"], name="Recall", mode="lines+markers", line=dict(color="#10B981", width=2)))
    fig.add_trace(go.Scatter(x=hist["days_of_history"], y=hist["f1"], name="F1 Score", mode="lines+markers", line=dict(color="#F26A4B", width=3)))
    fig.update_layout(title="Model Accuracy vs. Days of History Available",
                      xaxis_title="Days of Training History", yaxis_title="Score", yaxis_range=[0, 1.05])
    return _style(fig, 280)

