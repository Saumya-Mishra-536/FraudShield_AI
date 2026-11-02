"""FraudShield AI - visual theme for the Streamlit dashboard.

Dark navy canvas, electric-blue accent, glass panels. Everything visual lives
here so dashboard.py stays about content. Call inject() once, then use hero(),
section() and stat_strip() for the repeated building blocks.
"""
from __future__ import annotations

import html

import matplotlib as mpl
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

INK = "#050a1c"
PANEL = "#0a1330"
LINE = "rgba(120,150,255,.16)"
BLUE = "#2f6bff"
SKY = "#6aa0ff"
TEXT = "#e6ecff"
MUTED = "#93a2d4"
HEAT = LinearSegmentedColormap.from_list("fs_heat", ["#0b1638", "#1d3fa8", "#2f6bff", "#8fb4ff"])

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp {
  background:
    radial-gradient(900px 480px at 82% -6%, rgba(47,107,255,.30), transparent 62%),
    radial-gradient(700px 420px at -8% 18%, rgba(30,64,175,.22), transparent 60%),
    #050a1c;
  color: #e6ecff;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1240px; padding-top: 1.4rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { letter-spacing: -0.02em; font-weight: 600; color: #f4f7ff; }
[data-testid="stSidebar"] { background: #070d24; border-right: 1px solid rgba(120,150,255,.12); }
[data-testid="stSidebar"] hr { border-color: rgba(120,150,255,.12); }

/* hero */
.fs-hero { display: grid; grid-template-columns: 1.15fr .85fr; gap: 2rem; align-items: center; padding: 2.2rem 0 1.4rem; }
.fs-pill { display: inline-flex; align-items: center; gap: .5rem; font-size: .8rem; color: #b8c7ff;
  background: rgba(47,107,255,.12); border: 1px solid rgba(106,160,255,.28); padding: .3rem .8rem; border-radius: 999px; }
.fs-pill i { width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 10px #34d399; }
.fs-h1 { font-size: clamp(2.3rem, 5vw, 4rem); line-height: 1.04; font-weight: 600; letter-spacing: -0.035em; margin: 1.1rem 0 1rem; color: #f7f9ff; }
.fs-h1 span { display: block; background: linear-gradient(90deg, #3b7bff, #7db0ff); -webkit-background-clip: text; background-clip: text; color: transparent; }
.fs-sub { color: #93a2d4; font-size: 1.05rem; line-height: 1.6; max-width: 34rem; margin: 0; }
.fs-stack { position: relative; height: 290px; perspective: 900px; }
.fs-card { position: absolute; inset: 0 12% 0 8%; border-radius: 22px; border: 1px solid rgba(140,175,255,.35);
  background: linear-gradient(145deg, rgba(60,110,255,.34), rgba(10,19,48,.55)); backdrop-filter: blur(6px);
  box-shadow: 0 20px 60px rgba(20,60,200,.35), inset 0 1px 0 rgba(255,255,255,.18); }
.fs-card.c1 { transform: translate(34px,-10px) rotate(9deg) scale(.9); opacity: .38; }
.fs-card.c2 { transform: translate(14px,-4px) rotate(4.5deg) scale(.95); opacity: .62; }
.fs-card.c3 { padding: 1.5rem 1.6rem; display: flex; flex-direction: column; justify-content: space-between;
  background: linear-gradient(150deg, rgba(47,107,255,.5), rgba(8,16,44,.82)); }
.fs-card small { color: #b8c7ff; font-size: .8rem; }
.fs-card b { display: block; font-size: 2.7rem; font-weight: 600; letter-spacing: -0.03em; color: #fff; line-height: 1.1; }
.fs-card em { font-style: normal; color: #7fe3b5; font-size: .85rem; }

/* stat strip */
.fs-stats { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid rgba(120,150,255,.14);
  border-bottom: 1px solid rgba(120,150,255,.14); margin: .8rem 0 1.6rem; }
.fs-stat { padding: 1.3rem 1.4rem; border-left: 1px solid rgba(120,150,255,.14); }
.fs-stat:first-child { border-left: 0; padding-left: 0; }
.fs-stat b { display: block; font-size: 2.1rem; font-weight: 600; letter-spacing: -0.03em; color: #fff; }
.fs-stat span { color: #93a2d4; font-size: .9rem; }

.fs-note { font-size: .84rem; color: #a9b6e6; background: rgba(255,255,255,.03); border: 1px solid rgba(120,150,255,.14);
  border-radius: 14px; padding: .8rem 1.1rem; margin: 0 0 1.4rem; line-height: 1.55; }
.fs-note b { color: #e6ecff; }
.fs-section { margin: 1.6rem 0 .6rem; }
.fs-section h2 { font-size: 1.7rem; margin: 0 0 .2rem; }
.fs-section p { color: #93a2d4; margin: 0; }

/* top navigation (container key="nav") */
.st-key-nav div[role="radiogroup"] { gap: .3rem; background: rgba(255,255,255,.04); border: 1px solid rgba(120,150,255,.16);
  border-radius: 999px; padding: .3rem; width: fit-content; max-width: 100%; flex-wrap: wrap; }
.st-key-nav label { padding: .45rem 1.1rem; border-radius: 999px; margin: 0; cursor: pointer; transition: background .15s; }
.st-key-nav label > div:first-child { display: none; }
.st-key-nav label p { font-size: .9rem; color: #b8c7ff; margin: 0; }
.st-key-nav label:has(input:checked) { background: #2f6bff; box-shadow: 0 6px 22px rgba(47,107,255,.4); }
.st-key-nav label:has(input:checked) p { color: #fff; font-weight: 600; }
.st-key-nav label:focus-within { outline: 2px solid #8fb4ff; outline-offset: 2px; }

/* components */
[data-testid="stMetric"] { background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
  border: 1px solid rgba(120,150,255,.16); border-radius: 18px; padding: 1rem 1.2rem; }
[data-testid="stMetricLabel"] p { color: #93a2d4; font-size: .85rem; }
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; }
[data-testid="stAlert"] { background: rgba(47,107,255,.08); border: 1px solid rgba(120,150,255,.2); border-radius: 16px; color: #dbe4ff; }
[data-testid="stExpander"] { background: rgba(255,255,255,.025); border: 1px solid rgba(120,150,255,.16); border-radius: 16px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(120,150,255,.14); }
[data-testid="stImage"] img { background: #fff; border-radius: 16px; padding: 10px; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid rgba(120,150,255,.14); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; color: #93a2d4; }
.stTabs [aria-selected="true"] { color: #fff; }
.stButton > button[kind="primary"] { background: #2f6bff; border: 0; border-radius: 12px; font-weight: 600;
  padding: .6rem 1.4rem; box-shadow: 0 10px 30px rgba(47,107,255,.38); transition: transform .12s, box-shadow .12s; }
.stButton > button[kind="primary"]:hover { background: #4a80ff; transform: translateY(-1px); }
.stNumberInput input, .stSelectbox [data-baseweb="select"] > div { background: rgba(255,255,255,.04); border-radius: 12px; }

/* decision badges */
.badge-allow, .badge-challenge, .badge-block, .badge-hard-block { display: inline-block; padding: .35rem .95rem; border-radius: 999px; font-weight: 600; font-size: .95rem; }
.badge-allow { background: rgba(52,211,153,.14); color: #6ee7b7; border: 1px solid rgba(52,211,153,.4); }
.badge-challenge { background: rgba(251,191,36,.12); color: #fcd34d; border: 1px solid rgba(251,191,36,.4); }
.badge-block { background: rgba(248,113,113,.13); color: #fca5a5; border: 1px solid rgba(248,113,113,.45); }
.badge-hard-block { background: #b91c1c; color: #fff; border: 1px solid #ef4444; }

@media (max-width: 860px) {
  .fs-hero { grid-template-columns: 1fr; } .fs-stack { height: 220px; }
  .fs-stats { grid-template-columns: 1fr 1fr; } .fs-stat:nth-child(3) { border-left: 0; padding-left: 0; }
}
"""


def inject() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def mpl_dark() -> None:
    """Make matplotlib charts sit on the dark canvas (transparent background)."""
    mpl.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True, "text.color": TEXT, "axes.labelcolor": MUTED,
        "axes.edgecolor": "#24305e", "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#24305e", "axes.titlecolor": "#f4f7ff", "legend.facecolor": "#0a1330",
        "legend.edgecolor": "#24305e", "legend.labelcolor": TEXT,
    })


def hero(metrics: dict | None, engine=None) -> None:
    a = (metrics or {}).get("asymmetric_model", {})
    s = (metrics or {}).get("symmetric_model", {})
    tp, fn = a.get("true_positives"), a.get("false_negatives")
    caught = f"{tp} of {tp + fn}" if tp is not None else "-"
    tau = f"{engine.threshold:.3f}" if engine else "-"
    html_ = (
        '<div class="fs-hero"><div>'
        '<span class="fs-pill"><i></i>Decision support for fraud teams</span>'
        '<h1 class="fs-h1">Catch more fraud.<span>Block fewer customers.</span></h1>'
        '<p class="fs-sub">FraudShield prices every decision. It trains on the cost of a missed fraud versus a '
        'needless alert, calibrates its scores, and explains each flag in plain terms.</p></div>'
        '<div class="fs-stack"><div class="fs-card c1"></div><div class="fs-card c2"></div>'
        '<div class="fs-card c3"><small>Frauds caught on held-out test</small>'
        f'<div><b>{caught}</b><em>Decision cutoff {tau}</em></div></div></div></div>'
    )
    st.markdown(html_, unsafe_allow_html=True)
    if a and s:
        saving = (s["total_asymmetric_cost"] - a["total_asymmetric_cost"]) / s["total_asymmetric_cost"]
        stat_strip([
            (f"{a['recall']:.1%}", "Fraud caught"),
            (f"{saving:.0%}", "Lower cost than baseline"),
            (f"{a['average_precision_pr_auc']:.3f}", "Precision-recall AUC"),
            (f"{a['brier_score']:.4f}", "Brier score, calibrated"),
        ])


def stat_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="fs-stat"><b>{html.escape(v)}</b><span>{html.escape(l)}</span></div>' for v, l in items
    )
    st.markdown(f'<div class="fs-stats">{cells}</div>', unsafe_allow_html=True)


def note(text_html: str) -> None:
    st.markdown(f'<div class="fs-note">{text_html}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    sub_html = f"<p>{html.escape(sub)}</p>" if sub else ""
    st.markdown(f'<div class="fs-section"><h2>{html.escape(title)}</h2>{sub_html}</div>', unsafe_allow_html=True)
