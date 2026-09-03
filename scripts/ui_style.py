"""
scripts/ui_style.py
====================
Shared visual style for all dashboard pages.

Usage in any page::

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ui_style import inject_css, page_header, info_card, metric_row

The CSS matches the home page (app.py) so every page feels like the same app.
"""
import streamlit as st

# ── Palette (mirrors app.py) ────────────────────────────────────────────────
C_BLUE   = "#378ADD"
C_TEAL   = "#1D9E75"
C_PURPLE = "#7F77DD"
C_CORAL  = "#E85D3A"
C_BONE   = "#C8BFA9"

_CSS = """
<style>
/* Wider content area */
.block-container {
    max-width: 1100px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

/* Page header */
.page-header { margin-bottom: 1.4rem; }
.page-label  {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.2rem;
}
.page-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #1C1C2E;
    margin: 0 0 0.25rem 0;
    line-height: 1.2;
}
.page-sub {
    font-size: 0.9rem;
    color: #666;
    margin: 0;
}

/* Section label */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.5rem;
    margin-top: 0.2rem;
}

/* Generic card */
.ui-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.25rem 1.2rem;
    border-top: 4px solid var(--card-color, #378ADD);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    height: 100%;
}
.ui-card:hover { box-shadow: 0 6px 22px rgba(0,0,0,0.11); }

/* Info / highlight card */
.info-card {
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    border-left: 5px solid var(--info-color, #378ADD);
    margin-bottom: 1rem;
    background: var(--info-bg, #EEF6FF);
}
.info-card p { margin: 0; font-size: 0.88rem; line-height: 1.6; color: #1C1C2E; }

/* Metric card */
.metric-card {
    background: #fff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 3px solid var(--m-color, #378ADD);
}
.metric-val {
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--m-color, #378ADD);
}
.metric-lab {
    font-size: 0.72rem;
    color: #888;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Status badge */
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 0.18rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-green  { background: #DCFCE7; color: #166534; }
.badge-blue   { background: #DBEAFE; color: #1e40af; }
.badge-purple { background: #EDE9FE; color: #5B21B6; }
.badge-amber  { background: #FEF3C7; color: #92400E; }
.badge-red    { background: #FEE2E2; color: #991B1B; }

/* Chip */
.chip {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA;
    border-radius: 20px;
    padding: 0.18rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.1rem 0.08rem;
}
</style>
"""


def inject_css() -> None:
    """Call once per page after set_page_config to inject shared CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: str = "",
    label: str = "",
    color: str = C_BLUE,
) -> None:
    """
    Render a clean, consistent page header.

    Parameters
    ----------
    title    : Main heading text.
    subtitle : Gray caption below the title.
    label    : Small ALL-CAPS label above the title (e.g. "Stage 2 · Generator").
    color    : Accent colour for the left rule (hex string).
    """
    label_html = (
        f'<p class="page-label">{label}</p>' if label else ""
    )
    sub_html = (
        f'<p class="page-sub">{subtitle}</p>' if subtitle else ""
    )
    st.markdown(
        f"""
<div class="page-header"
     style="border-left:4px solid {color}; padding-left:1rem;">
    {label_html}
    <h1 class="page-title">{title}</h1>
    {sub_html}
</div>
""",
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Small ALL-CAPS divider label before a content section."""
    st.markdown(
        f'<p class="section-label">{text}</p>',
        unsafe_allow_html=True,
    )


def info_card(
    body: str,
    color: str = C_BLUE,
    bg: str = "",
    title: str = "",
) -> None:
    """
    Render a left-bordered highlight card.

    Parameters
    ----------
    body  : HTML or plain text content.
    color : Border colour (hex).
    bg    : Background colour (hex). Defaults to a tint of `color`.
    title : Optional bold title above the body.
    """
    if not bg:
        bg = "#EEF6FF" if color == C_BLUE else "#F3F1FF" if color == C_PURPLE \
            else "#EDFAF4" if color == C_TEAL else "#FEF3F2"
    title_html = (
        f'<p style="font-weight:700;font-size:0.85rem;margin:0 0 0.4rem;">{title}</p>'
        if title else ""
    )
    st.markdown(
        f"""
<div class="info-card"
     style="--info-color:{color}; --info-bg:{bg}; background:{bg};">
    {title_html}
    <p>{body}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[tuple[str, str, str]]) -> None:
    """
    Render a row of metric cards.

    Parameters
    ----------
    metrics : List of (value, label, color) tuples.
    """
    cols = st.columns(len(metrics), gap="medium")
    for col, (val, lab, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
<div class="metric-card" style="--m-color:{color}">
    <div class="metric-val">{val}</div>
    <div class="metric-lab">{lab}</div>
</div>""",
                unsafe_allow_html=True,
            )


def badge(text: str, style: str = "blue") -> str:
    """Return an inline HTML badge string. style: green/blue/purple/amber/red."""
    return f'<span class="badge badge-{style}">{text}</span>'


def sidebar_nav_sections() -> None:
    """
    Inject visual section grouping labels into the sidebar.
    Call from app.py (or any page) once per run.
    The labels appear above the auto-generated page links using
    st.sidebar.markdown with zero-height spacers — no st.navigation
    required, so it works on any Streamlit version.
    """
    st.sidebar.markdown(
        """
<style>
/* ── Sidebar section headers ── */
.nav-section {
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #aaa;
    padding: 0.55rem 0 0.1rem 0.2rem;
    border-top: 1px solid #e5e7ec;
    margin-top: 0.4rem;
}
.nav-section:first-child { border-top: none; margin-top: 0; }
</style>
""",
        unsafe_allow_html=True,
    )


def qic_pipeline_sidebar() -> None:
    """
    Persistent QIC journey tracker rendered in every page sidebar.
    Reads st.session_state to colour each step live.
    Call once per page after inject_css().
    """
    import streamlit as st

    ss = st.session_state

    # ── step definitions (label, session-state key or callable) ──
    steps = [
        ("📂 Data loaded",
         lambda s: "real_volume" in s or "d2im_scan" in s or "bone_volume" in s),
        ("🔍 ROI detected",
         lambda s: "real_bone_mask" in s or "real_bone_mask_trabecular" in s),
        ("🧬 Volume generated",
         lambda s: "bone_volume" in s),
        ("⚙️  FE solved",
         lambda s: "pipeline_fe" in s),
        ("📐 Strain registered",
         lambda s: s.get("strain_registered", False)),
        ("🧊 3D visualised",
         lambda s: "mesh_verts" in s),
        ("⚛️  QRC features",
         lambda s: "qrc_features" in s or "qrc_accuracy" in s),
        ("🔬 QIC ready",
         lambda s: ("pipeline_fe" in s or "qrc_accuracy" in s)
                   and ("bone_volume" in s or "real_volume" in s)),
    ]

    done_count = sum(1 for _, fn in steps if fn(ss))

    st.sidebar.markdown("---")

    # ── compact progress bar ──
    pct = int(done_count / len(steps) * 100)
    bar_fill = "#1D9E75" if pct == 100 else "#378ADD"
    st.sidebar.markdown(
        f"""<div style="margin-bottom:6px">
  <div style="display:flex;justify-content:space-between;
              font-size:0.62rem;color:#888;margin-bottom:3px">
    <span style="font-weight:700;letter-spacing:0.1em;text-transform:uppercase">
      QIC Journey</span>
    <span style="color:{bar_fill};font-weight:600">{pct}%</span>
  </div>
  <div style="background:#e5e7eb;border-radius:4px;height:5px;overflow:hidden">
    <div style="background:{bar_fill};width:{pct}%;height:5px;
                border-radius:4px;transition:width .4s"></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── step list ──
    rows_html = ""
    for label, fn in steps:
        done = fn(ss)
        dot  = f'<span style="color:#1D9E75;font-size:0.85rem">●</span>' if done                else f'<span style="color:#d1d5db;font-size:0.85rem">○</span>'
        txt_col = "#374151" if done else "#9ca3af"
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:7px;'
            f'padding:3px 0;font-size:0.76rem;color:{txt_col}">'
            f'{dot} {label}</div>'
        )

    st.sidebar.markdown(
        f'<div style="line-height:1.5">{rows_html}</div>',
        unsafe_allow_html=True,
    )

    # ── what to do next ──
    next_labels = [lbl for lbl, fn in steps if not fn(ss)]
    if next_labels:
        nxt = next_labels[0]
        st.sidebar.markdown(
            f'<div style="margin-top:6px;font-size:0.68rem;color:#6b7280">'
            f'▶ Next: <span style="color:#378ADD;font-weight:600">{nxt}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.success("🎉 QIC pipeline complete!")

    # ── compact workflow guide ──
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """<div style="font-size:0.60rem;font-weight:800;letter-spacing:0.14em;
text-transform:uppercase;color:#9ca3af;margin-bottom:4px">Workflow</div>
<div style="font-size:0.73rem;color:#6b7280;line-height:2">
<span style="color:#9ca3af;font-weight:700">① Prepare</span>
  📂 Data &nbsp;·&nbsp; 🔍 ROI<br>
<span style="color:#9ca3af;font-weight:700">② Run</span>
  🧬 Gen &nbsp;·&nbsp; ⚙️ FE &nbsp;·&nbsp; 🔗 Pipeline<br>
<span style="color:#9ca3af;font-weight:700">③ Analyse</span>
  ⚛️ QRC &nbsp;·&nbsp; 🔬 QKSVM &nbsp;·&nbsp; 🧽 Foam
</div>""",
        unsafe_allow_html=True,
    )
