"""SyncGuard AI Design System: Cyber Analytics / Dark Mode.
Futuristic, high-contrast, obsidian dark theme with glowing neon accents.
Inspired by mission-control telemetry, Linear, and Datadog Dark.
"""
import base64
import os
import re
import streamlit as st

# ================= COLOR TOKENS =================
# Backgrounds & Surfaces (Obsidian Scale)
BG = "#0A0D14"               # Deep obsidian canvas
BG_SIDEBAR = "#0E121C"       # Secondary dark pane
SURFACE = "#131826"          # Primary card surface
SURFACE_ELEVATED = "#192033" # Hover / elevated surface
BORDER = "#1E2638"           # Default dark borders
BORDER_GLOW = "rgba(0, 242, 254, 0.3)"
BORDER_STRONG = "#2B364F"

# Typography
TEXT_MAIN = "#F0F4FC"        # Crisp white headers
TEXT_BODY = "#A2B0C7"        # High-contrast slate body
TEXT_MUTED = "#687794"       # Subtitles & telemetry captions

# Cyber Accent Palette
PRIMARY = "#00F2FE"          # Electric Cyan
PRIMARY_GLOW = "rgba(0, 242, 254, 0.15)"
CYBER_VIOLET = "#8A2BE2"     # Electric Violet
CYBER_BLUE = "#3B82F6"       # Deep Neon Blue

# High-Contrast Semantic Colors (with glow values)
SUCCESS = "#00E599"          # Electric Emerald
SUCCESS_BG = "rgba(0, 229, 153, 0.12)"
SUCCESS_BORDER = "rgba(0, 229, 153, 0.3)"

WARNING = "#FFB800"          # Neon Amber
WARNING_BG = "rgba(255, 184, 0, 0.12)"
WARNING_BORDER = "rgba(255, 184, 0, 0.3)"

DANGER = "#FF3B5C"           # Laser Red
DANGER_BG = "rgba(255, 59, 92, 0.14)"
DANGER_BORDER = "rgba(255, 59, 92, 0.35)"

NEUTRAL = "#687794"
NEUTRAL_BG = "rgba(104, 119, 148, 0.14)"
NEUTRAL_BORDER = "rgba(104, 119, 148, 0.28)"

SEV_COLORS = {
    "High": {"color": DANGER, "bg": DANGER_BG, "border": DANGER_BORDER, "icon": "⚡"},
    "Medium": {"color": WARNING, "bg": WARNING_BG, "border": WARNING_BORDER, "icon": "▲"},
    "Low": {"color": PRIMARY, "bg": PRIMARY_GLOW, "border": "rgba(0, 242, 254, 0.25)", "icon": "●"},
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global Font & Canvas */
html, body, .stApp, p, label, .stMarkdown, div[data-testid="stMarkdownContainer"], button, input, select, textarea {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: {TEXT_BODY};
    -webkit-font-smoothing: antialiased;
}}

/* Main Background */
.stApp {{
    background-color: {BG} !important;
}}

/* Top Padding & Layout */
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px;
}}

/* Sidebar - Deep Telemetry Deck */
section[data-testid="stSidebar"] {{
    background-color: {BG_SIDEBAR} !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}}

/* Headings */
h1, h2, h3, h4 {{
    font-family: 'Inter', sans-serif !important;
    color: {TEXT_MAIN} !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}}

/* Top HUD Banner */
.sg-hud {{
    background: linear-gradient(135deg, rgba(19, 24, 38, 0.95) 0%, rgba(14, 18, 28, 0.95) 100%);
    border: 1px solid {BORDER};
    border-left: 4px solid {PRIMARY};
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}}
.sg-hud h2 {{
    margin: 0;
    font-size: 1.2rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    display: flex;
    align-items: center;
    gap: 8px;
}}
.sg-hud p {{
    margin: 4px 0 0 0;
    font-size: 0.85rem;
    color: {TEXT_MUTED};
}}

/* Cyber Metric Card */
.sg-kpi {{
    background: linear-gradient(145deg, rgba(25, 32, 51, 0.8) 0%, rgba(19, 24, 38, 0.95) 100%);
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 18px;
    min-height: 106px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}}
.sg-kpi:hover {{
    border-color: {PRIMARY};
    box-shadow: 0 0 18px rgba(0, 242, 254, 0.15), 0 4px 16px rgba(0, 0, 0, 0.4);
    transform: translateY(-1px);
}}
.sg-kpi .sg-kpi-label {{
    font-size: 0.775rem;
    font-weight: 600;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}
.sg-kpi .sg-kpi-value {{
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.2;
    color: {TEXT_MAIN};
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.03em;
}}
.sg-kpi .sg-kpi-sub {{
    font-size: 0.775rem;
    color: {TEXT_MUTED};
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 5px;
}}

/* Card Container */
.sg-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}}

/* Section Header */
.sg-header {{
    margin-top: 1.75rem;
    margin-bottom: 0.875rem;
    display: flex;
    flex-direction: column;
    gap: 2px;
}}
.sg-header .sg-header-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    letter-spacing: -0.015em;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.sg-header .sg-header-title::before {{
    content: "";
    display: inline-block;
    width: 3px;
    height: 14px;
    background: {PRIMARY};
    border-radius: 2px;
    box-shadow: 0 0 8px {PRIMARY};
}}
.sg-header .sg-header-sub {{
    font-size: 0.8125rem;
    color: {TEXT_MUTED};
    margin-left: 11px;
}}

/* Cyber Badges / Pills */
.sg-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.775rem;
    font-weight: 600;
    line-height: 1.3;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}}
.sg-pill-high {{
    background: {DANGER_BG};
    color: {DANGER};
    border: 1px solid {DANGER_BORDER};
    box-shadow: 0 0 10px rgba(255, 59, 92, 0.25);
}}
.sg-pill-medium {{
    background: {WARNING_BG};
    color: {WARNING};
    border: 1px solid {WARNING_BORDER};
    box-shadow: 0 0 10px rgba(255, 184, 0, 0.2);
}}
.sg-pill-low {{
    background: {PRIMARY_GLOW};
    color: {PRIMARY};
    border: 1px solid rgba(0, 242, 254, 0.3);
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
}}
.sg-pill-success {{
    background: {SUCCESS_BG};
    color: {SUCCESS};
    border: 1px solid {SUCCESS_BORDER};
    box-shadow: 0 0 10px rgba(0, 229, 153, 0.2);
}}

/* Cyber Callout */
.sg-tip {{
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.06) 0%, rgba(19, 24, 38, 0.8) 100%);
    border: 1px solid rgba(0, 242, 254, 0.25);
    color: {TEXT_MAIN};
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    box-shadow: 0 0 14px rgba(0, 242, 254, 0.06);
}}
.sg-tip code {{
    background: rgba(0, 242, 254, 0.12) !important;
    color: {PRIMARY} !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
}}

/* Empty State */
.sg-empty {{
    background: {SURFACE};
    border: 1px dashed {BORDER_STRONG};
    border-radius: 10px;
    padding: 40px 24px;
    text-align: center;
    margin: 1.5rem 0;
}}
.sg-empty-icon {{
    font-size: 2.2rem;
    color: {PRIMARY};
    margin-bottom: 8px;
    text-shadow: 0 0 15px {PRIMARY};
}}
.sg-empty-title {{
    font-size: 1.05rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    margin-bottom: 4px;
}}
.sg-empty-desc {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    max-width: 440px;
    margin: 0 auto;
}}

/* Sidebar Radio Navigation Buttons (Modern App Sidebar) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
    gap: 4px !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
    background: transparent !important;
    border: 1px solid transparent !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: {TEXT_BODY} !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
    background: rgba(255, 255, 255, 0.04) !important;
    color: {TEXT_MAIN} !important;
    border-color: {BORDER} !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] input:checked + div {{
    background: rgba(0, 242, 254, 0.1) !important;
    border: 1px solid rgba(0, 242, 254, 0.35) !important;
    color: {PRIMARY} !important;
    box-shadow: 0 0 12px rgba(0, 242, 254, 0.12) !important;
}}

/* Expanders */
div[data-testid="stExpander"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    margin-bottom: 0.75rem;
}}
div[data-testid="stExpander"] details {{
    border-radius: 10px !important;
}}
div[data-testid="stExpander"] summary {{
    padding: 10px 14px !important;
    font-weight: 500 !important;
    color: {TEXT_MAIN} !important;
}}
div[data-testid="stExpander"] summary:hover {{
    background-color: {SURFACE_ELEVATED} !important;
}}

/* Buttons */
.stButton > button {{
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    border: 1px solid {BORDER_STRONG} !important;
    background: {SURFACE} !important;
    color: {TEXT_MAIN} !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
    border-color: {PRIMARY} !important;
    color: {PRIMARY} !important;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.2) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 0 15px rgba(0, 198, 255, 0.35) !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 22px rgba(0, 198, 255, 0.6) !important;
    transform: translateY(-1px) !important;
}}

/* Dataframe Styling */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {SURFACE};
}}

/* Hide helper promo toast / popovers */
[data-testid="stFloatingToastContainer"],
.stFloatingToastContainer,
#MainMenu, footer, header {{
    visibility: hidden;
    height: 0;
}}
</style>
"""

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _logo_file():
    for name in ("logo.png", "logo.jpg", "logo.jpeg"):
        p = os.path.join(ASSETS, name)
        if os.path.exists(p):
            return p
    return None


def logo_html(size=36):
    p = _logo_file()
    if p:
        ext = "png" if p.endswith("png") else "jpeg"
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        return f'<img src="data:image/{ext};base64,{b64}" width="{size}" height="{size}" style="border-radius:8px;vertical-align:middle;box-shadow:0 0 10px rgba(0,242,254,0.3);">'
    svg_path = os.path.join(ASSETS, "logo.svg")
    if os.path.exists(svg_path):
        svg = open(svg_path).read()
        return f'<div style="width:{size}px;height:{size}px;display:inline-block;vertical-align:middle;">{svg}</div>'.replace(
            'width="64" height="64"', f'width="{size}" height="{size}"'
        )
    return f'<div style="width:{size}px;height:{size}px;background:{PRIMARY_GLOW};border:1px solid {PRIMARY};border-radius:8px;display:flex;align-items:center;justify-content:center;color:{PRIMARY};font-weight:700;font-size:14px;box-shadow:0 0 12px {PRIMARY};">SG</div>'


def page_icon():
    return _logo_file() or "⚡"


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def status_headline(headline, subtext="", badge_text=None, badge_type="high"):
    badge_html = ""
    if badge_text:
        badge_html = pill(badge_type.capitalize(), badge_text)
    sub_html = f"<p>{subtext}</p>" if subtext else ""
    st.markdown(
        f'<div class="sg-hud">'
        f'<div>'
        f'<h2>{headline}</h2>'
        f'{sub_html}'
        f'</div>'
        f'{badge_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:4px 0 16px 0;border-bottom:1px solid {BORDER};margin-bottom:1.5rem;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'{logo_html(40)}'
        f'<div>'
        f'<h1 style="margin:0;font-size:1.45rem;font-weight:700;color:{TEXT_MAIN};line-height:1.2;letter-spacing:-0.02em;">{title}</h1>'
        f'<p style="margin:2px 0 0 0;font-size:0.85rem;color:{TEXT_MUTED};">{subtitle}</p>'
        f'</div>'
        f'</div>'
        f'<div style="display:inline-flex;align-items:center;gap:8px;background:{SURFACE};border:1px solid {BORDER};padding:6px 12px;border-radius:20px;font-size:0.75rem;font-family:monospace;color:{SUCCESS};">'
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{SUCCESS};box-shadow:0 0 8px {SUCCESS};"></span>'
        f'SYSTEM ONLINE · 12ms'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, sub="", status="neutral", color=None):
    """Cyber metric card component."""
    sub_html = f'<div class="sg-kpi-sub">{sub}</div>' if sub else ""
    val_color = TEXT_MAIN
    if status == "danger" or color == DANGER:
        val_color = DANGER
    elif status == "warning" or color == WARNING:
        val_color = WARNING
    elif status == "success" or color == SUCCESS:
        val_color = SUCCESS

    col.markdown(
        f'<div class="sg-kpi">'
        f'<div>'
        f'<div class="sg-kpi-label">{label}</div>'
        f'<div class="sg-kpi-value" style="color:{val_color};">{value}</div>'
        f'</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def card(content_html):
    """Card wrapper component."""
    return f'<div class="sg-card">{content_html}</div>'


def section(title, sub=""):
    """Section header component with 1-line helper text."""
    sub_html = f'<div class="sg-header-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="sg-header">'
        f'<div class="sg-header-title">{title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def empty_state(title, message, icon="⚡"):
    """Empty state component."""
    st.markdown(
        f'<div class="sg-empty">'
        f'<div class="sg-empty-icon">{icon}</div>'
        f'<div class="sg-empty-title">{title}</div>'
        f'<div class="sg-empty-desc">{message}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def tip(text):
    """Contextual callout component."""
    rendered = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    rendered = re.sub(r'`(.*?)`', r'<code>\1</code>', rendered)
    st.markdown(f'<div class="sg-tip"><span style="flex-shrink:0;">⚡</span><div>{rendered}</div></div>', unsafe_allow_html=True)


def pill(severity, label=None):
    """Glowing neon badge component."""
    sev = severity.capitalize() if isinstance(severity, str) else "Low"
    cfg = SEV_COLORS.get(sev, SEV_COLORS["Low"])
    display_label = label or sev
    cls_name = f"sg-pill-{sev.lower()}"
    return (
        f'<span class="sg-pill {cls_name}">'
        f'<span style="font-size:0.65rem;">{cfg["icon"]}</span> '
        f'{display_label}'
        f'</span>'
    )
