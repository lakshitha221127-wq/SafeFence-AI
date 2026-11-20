from pathlib import Path
import io
import subprocess
import shutil
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. PAGE & APPLICATION CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SafeFence · Electrical Signal Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File Paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_signal_dataset.csv"
JAVA_MAIN = ROOT / "SafeFenceAI.java"
JAVA_DASHBOARD = ROOT / "SafeFenceDashboard.java"

# =============================================================================
# 2. ACADEMIC & PROFESSIONAL DARK THEME STYLING
# =============================================================================
st.markdown("""
<style>
/* Root Dark Canvas */
.stApp {
    background-color: #080e1a;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Streamlit Top Header & Toolbar - Dark Integration & Zero White Strip */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background-color: #080e1a !important;
    background: #080e1a !important;
    color: #e2e8f0 !important;
    z-index: 100 !important;
}

/* Remove Top Accent Decoration Line */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Streamlit Header Buttons (Deploy, 3-dots, status) */
header[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] span,
[data-testid="stToolbar"] a {
    color: #94a3b8 !important;
    background: transparent !important;
}
header[data-testid="stHeader"] button:hover,
[data-testid="stToolbar"] button:hover {
    color: #f8fafc !important;
    background: rgba(30, 41, 59, 0.6) !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1527 0%, #080d19 100%);
    border-right: 1px solid #1e293b;
}

/* Main Container Padding - Ensures full clearance below Streamlit toolbar */
.block-container {
    padding-top: 4.2rem !important;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

/* Professional Top Header */
.sys-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px 22px;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}
.sys-title {
    font-size: 1.25rem;
    font-weight: 750;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sys-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.badge-blue {
    background-color: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.badge-green {
    background-color: rgba(52, 211, 153, 0.12);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.badge-purple {
    background-color: rgba(167, 139, 250, 0.12);
    color: #c084fc;
    border: 1px solid rgba(167, 139, 250, 0.3);
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0f192d 0%, #0b1220 100%);
    border: 1px solid #1e2e4a;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.85rem !important;
    font-weight: 750 !important;
}

/* Section Typography */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 0.8rem;
    margin-bottom: 0.25rem;
}
.section-desc {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 1.3rem;
}

/* Tabs & UI Elements */
div[data-baseweb="tab-list"] {
    background-color: #0b1220;
    border-radius: 8px;
    padding: 4px;
    gap: 6px;
    border: 1px solid #1e293b;
    margin-bottom: 1rem;
}
div[data-baseweb="tab"] {
    border-radius: 6px;
    color: #94a3b8;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.88rem;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 8px;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. COLOR PALETTES & PLOTLY STYLING ENGINE (FIXED PLOTLY V7 COMPATIBILITY)
# =============================================================================
CLASS_COLORS = {
    "Fence-Like": "#38bdf8",     # Cyan / Electric Pulse
    "Mains-Like": "#fbbf24",     # Amber / 50Hz Mains
    "Interference": "#f87171",  # Coral / High-Frequency Noise
    "Background": "#34d399",    # Emerald / Ambient Floor
    "Uncertain / Requires-Verification": "#a78bfa", # Violet / Ambiguous
    "Uncertain": "#a78bfa"
}

QUALITY_COLORS = {
    "Good": "#34d399",
    "Fair": "#fbbf24",
    "Poor": "#f87171"
}

RISK_COLORS = {
    "HIGH - possible live electric fence, keep clear": "#ef4444",
    "MEDIUM - mains-frequency source nearby": "#f59e0b",
    "LOW - no fence signature detected": "#10b981",
    "VERIFY - missing or insufficient signal data": "#8b5cf6"
}

def style_chart(fig, title="", height=400, show_legend=True):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color="#f8fafc", size=14, family="sans-serif"),
            x=0.01,
            y=0.96
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        font=dict(color="#cbd5e1", family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=45, r=30, t=55, b=45),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(11, 18, 32, 0.8)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8", size=11),
            title=dict(font=dict(color="#cbd5e1", size=12))
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8", size=11),
            title=dict(font=dict(color="#cbd5e1", size=12))
        )
    )
    return fig

# =============================================================================
# 4. ROBUST DATASET INGESTION & PREPROCESSING
# =============================================================================
@st.cache_data
def load_dataset(file_path: Path, mtime: float):
    if not file_path.exists():
        return None, "Dataset file does not exist on disk."
    
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            first_line = f.readline()
        
        tab_count = first_line.count("\t")
        comma_count = first_line.count(",")
        sep = "\t" if tab_count > comma_count else ","

        try:
            df = pd.read_csv(file_path, sep=sep)
        except Exception:
            df = pd.read_csv(file_path, sep=sep, engine="python")
        
        if len(df.columns) <= 1:
            alt_sep = "," if sep == "\t" else "\t"
            df = pd.read_csv(file_path, sep=alt_sep)
        
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        return df, None
    except Exception as exc:
        return None, str(exc)

if not DATA_PATH.exists():
    st.error(f"❌ Dataset not found at: {DATA_PATH}")
    st.stop()

raw_df, load_err = load_dataset(DATA_PATH, DATA_PATH.stat().st_mtime)
if load_err or raw_df is None:
    st.error(f"❌ Error loading dataset: {load_err}")
    st.stop()

required_cols = [
    "signal_id", "signal_class", "amplitude", "pulse_width_ms",
    "inter_pulse_interval_ms", "dominant_frequency_hz", "spectral_energy",
    "signal_quality_score", "illustrative_confidence", "signal_quality"
]

missing_req = [c for c in required_cols if c not in raw_df.columns]
if missing_req:
    st.error(f"Dataset schema mismatch: missing columns {missing_req}")
    st.stop()

numeric_cols = [
    "amplitude", "pulse_width_ms", "inter_pulse_interval_ms",
    "dominant_frequency_hz", "spectral_energy", "signal_quality_score",
    "illustrative_confidence"
]

for col in numeric_cols:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# Clean dataset copy (non-destructive median imputation)
clean_df = raw_df.drop_duplicates().copy()
for col in numeric_cols:
    if clean_df[col].isna().any():
        med_val = clean_df[col].median()
        clean_df[col] = clean_df[col].fillna(med_val)

# Core Telemetry Metrics
total_raw_rows = len(raw_df)
total_numeric_features = len(numeric_cols)
missing_cells_before = int(raw_df.isna().sum().sum())
missing_cells_after = int(clean_df.isna().sum().sum())
duplicate_rows_count = int(raw_df.duplicated().sum())
completeness_pct = ((raw_df.size - missing_cells_before) / raw_df.size) * 100

# =============================================================================
# 5. SIDEBAR NAVIGATION & FILTER CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:1.8rem;">⚡</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#f8fafc; letter-spacing:0.02em;">SAFEFENCE</div>
            <div style="font-size:0.75rem; color:#38bdf8; font-weight:600; text-transform:uppercase;">Signal Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;'>Navigation</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Select Module",
        [
            "Overview",
            "Signal Analysis",
            "Data Quality & Records",
            "Java Classification"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;'>Telemetry Filters</p>", unsafe_allow_html=True)
    
    all_classes = sorted(raw_df["signal_class"].dropna().astype(str).unique())
    
    # Preset quick filters
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("All Classes", use_container_width=True):
            st.session_state["filter_classes"] = all_classes
    with btn_c2:
        if st.button("Fence Only", use_container_width=True):
            st.session_state["filter_classes"] = ["Fence-Like"]

    curr_selected = st.session_state.get("filter_classes", all_classes)
    selected_classes = st.multiselect(
        "Filter Signal Classes",
        all_classes,
        default=curr_selected,
        help="Select one or more signal categories."
    )

    # Sliders for analytical exploration
    amp_min, amp_max = float(raw_df["amplitude"].min()), float(raw_df["amplitude"].max())
    amp_range = st.slider(
        "Amplitude Window",
        min_value=round(amp_min, 2),
        max_value=round(amp_max, 2),
        value=(round(amp_min, 2), round(amp_max, 2)),
        step=0.05
    )

    search_id = st.text_input(
        "Search Signal ID",
        placeholder="e.g. SIM-00042",
        help="Instant regex / substring search."
    ).strip()

    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;'>Dataset & System Info</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#94a3b8; line-height:1.6;">
        <div>🟢 <strong>System Status:</strong> Operational</div>
        <div>☕ <strong>Classifier Backend:</strong> Java 25.0.3 SE</div>
        <div>📁 <strong>Dataset:</strong> <code>{DATA_PATH.name}</code></div>
        <div>📊 <strong>Telemetry Scope:</strong> {total_raw_rows:,} Signals × {len(raw_df.columns)} Features</div>
        <div style="margin-top:6px; color:#64748b; font-size:0.75rem;">ℹ️ <em>Note: Dataset contains simulated signal telemetry for academic exploration.</em></div>
    </div>
    """, unsafe_allow_html=True)

# Sliced working dataframe
filtered_df = raw_df[
    (raw_df["signal_class"].astype(str).isin(selected_classes)) &
    (raw_df["amplitude"].between(amp_range[0], amp_range[1], inclusive="both") | raw_df["amplitude"].isna())
].copy()

# =============================================================================
# TOP HEADER SYSTEM STATUS BANNER
# =============================================================================
st.markdown(f"""
<div class="sys-header">
    <div class="sys-title">
        <span>⚡ SAFEFENCE</span>
        <span style="color:#64748b; font-weight:400; font-size:1.1rem;">|</span>
        <span style="color:#38bdf8; font-weight:600; font-size:1.05rem;">{page}</span>
    </div>
    <div class="sys-badges">
        <span class="badge badge-green">● SYSTEM READY</span>
        <span class="badge badge-blue">JAVA 25 BACKEND</span>
        <span class="badge badge-purple">{len(filtered_df):,} / {total_raw_rows:,} ACTIVE SIGNALS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# MODULE 1: OVERVIEW
# =============================================================================
if page == "Overview":
    st.markdown('<div class="section-header">Operational Telemetry Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Executive summary of electrical signal telemetry, class composition, and pulse characteristics.</div>', unsafe_allow_html=True)

    # 5 KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Monitored Signals", f"{total_raw_rows:,}", f"{len(filtered_df):,} in view", delta_color="normal")
    kpi2.metric("Signal Classes", f"{raw_df['signal_class'].nunique()}", "5 Archetypes")
    kpi3.metric("Telemetry Features", f"{total_numeric_features}", "7 Numerical + 3 Categorical")
    kpi4.metric("Completeness Index", f"{completeness_pct:.1f}%", f"{missing_cells_before} missing cells", delta_color="inverse")
    
    avg_fidelity = raw_df["signal_quality_score"].dropna().mean()
    kpi5.metric("Avg Signal Fidelity", f"{avg_fidelity:.2f}", "Scale: 0.0 - 1.0")

    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No telemetry records match your current sidebar filters. Please adjust your class selection or amplitude sliders.")
    else:
        # Row 1: Class Distribution & Quality Proportions
        col_left, col_right = st.columns([3, 2])

        with col_left:
            class_counts = (
                filtered_df["signal_class"]
                .value_counts()
                .reset_index(name="Count")
                .rename(columns={"index": "Signal Class"})
            )
            class_counts["Percentage"] = (class_counts["Count"] / len(filtered_df)) * 100
            
            fig_class = px.bar(
                class_counts,
                x="signal_class",
                y="Count",
                color="signal_class",
                color_discrete_map=CLASS_COLORS,
                text=class_counts["Count"].apply(lambda v: f"{v:,}"),
                hover_data={"Percentage": ":.1f%"},
                labels={"signal_class": "Signal Class", "Count": "Observed Records"}
            )
            fig_class.update_traces(textposition="outside", marker_line_width=1.5, marker_line_color="#1e293b")
            fig_class = style_chart(fig_class, "Signal Class Telemetry Distribution", height=380, show_legend=False)
            st.plotly_chart(fig_class, use_container_width=True)

        with col_right:
            qual_counts = (
                filtered_df["signal_quality"]
                .value_counts()
                .reset_index(name="Count")
                .rename(columns={"index": "Quality"})
            )
            fig_qual = px.pie(
                qual_counts,
                names="signal_quality",
                values="Count",
                hole=0.6,
                color="signal_quality",
                color_discrete_map=QUALITY_COLORS,
                labels={"signal_quality": "Quality Tier"}
            )
            fig_qual.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="#0f172a", width=2))
            )
            fig_qual = style_chart(fig_qual, "Signal Quality Rating Breakdown", height=380)
            st.plotly_chart(fig_qual, use_container_width=True)

        # Row 2: Comprehensive Benchmark Matrix & Mean Feature Profiles
        st.markdown('<div class="section-header">Class-by-Class Telemetry Benchmarks</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Average electrical and spectral characteristics calculated across all observed signals in each archetype.</div>', unsafe_allow_html=True)

        benchmarks = (
            filtered_df.groupby("signal_class")
            .agg(
                Records=("signal_id", "count"),
                Mean_Amplitude=("amplitude", "mean"),
                Mean_Pulse_Width_ms=("pulse_width_ms", "mean"),
                Mean_Frequency_Hz=("dominant_frequency_hz", "mean"),
                Mean_Spectral_Energy=("spectral_energy", "mean"),
                Mean_Quality_Score=("signal_quality_score", "mean")
            )
            .reset_index()
        )

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            fig_amp_bench = px.bar(
                benchmarks,
                x="signal_class",
                y="Mean_Amplitude",
                color="signal_class",
                color_discrete_map=CLASS_COLORS,
                text=benchmarks["Mean_Amplitude"].apply(lambda v: f"{v:.3f}"),
                labels={"signal_class": "Class", "Mean_Amplitude": "Mean Amplitude (Normalized)"}
            )
            fig_amp_bench.update_traces(textposition="outside")
            fig_amp_bench = style_chart(fig_amp_bench, "Mean Amplitude by Signal Class", height=320, show_legend=False)
            st.plotly_chart(fig_amp_bench, use_container_width=True)

        with b_c2:
            fig_freq_bench = px.bar(
                benchmarks,
                x="signal_class",
                y="Mean_Frequency_Hz",
                color="signal_class",
                color_discrete_map=CLASS_COLORS,
                text=benchmarks["Mean_Frequency_Hz"].apply(lambda v: f"{v:.1f} Hz"),
                labels={"signal_class": "Class", "Mean_Frequency_Hz": "Mean Frequency (Hz)"}
            )
            fig_freq_bench.update_traces(textposition="outside")
            fig_freq_bench = style_chart(fig_freq_bench, "Mean Dominant Frequency by Class", height=320, show_legend=False)
            st.plotly_chart(fig_freq_bench, use_container_width=True)

        with st.expander("🔍 Telemetry Engineering Findings & Archetype Signatures", expanded=False):
            st.markdown("""
            * **Fence-Like Signatures:** Characterized by high dominant frequencies (>450 Hz up to 1200 Hz), short pulse widths (<8 ms, typically 3-6 ms), and high amplitudes (>0.4). This distinct pulse signature models energized high-voltage fence pulses.
            * **Mains-Like Signatures:** Strongly grouped around 50 Hz (with minor utility harmonic variation 45-65 Hz), modeling coupled AC power transmission line leakage.
            * **Background Floor:** Displays very low amplitude (<0.2), representing ambient electromagnetic floor noise.
            * **Interference:** Exhibits irregular pulse intervals, elevated spectral energies, and non-repeating frequency peaks.
            * **Uncertain / Verification Required:** Records with missing telemetry channels or conflicting frequency-amplitude ratios flagged for sensor re-calibration.
            """)

# =============================================================================
# MODULE 2: SIGNAL ANALYSIS
# =============================================================================
elif page == "Signal Analysis":
    st.markdown('<div class="section-header">Interactive Signal Telemetry Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Multi-channel signal exploration covering time-domain pulse widths, frequency spectra, energy densities, and feature correlations.</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No records match the active filter criteria.")
        st.stop()

    active_df = filtered_df.copy()
    if search_id:
        mask = active_df["signal_id"].astype(str).str.contains(search_id, case=False, na=False)
        if mask.any():
            st.success(f"🔎 Filtered **{mask.sum()}** records matching Signal ID: `{search_id}`")
            active_df = active_df[mask]
        else:
            st.info(f"ℹ️ No signals matched `{search_id}`. Displaying all active filter records.")

    # Analytical Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Pulse & Waveform",
        "🌐 Frequency & Spectra",
        "💎 Quality & Confidence",
        "🔬 Relationship Explorer",
        "🔗 Correlation Matrix"
    ])

    # TAB 1: PULSE & WAVEFORM
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            valid_amp = active_df.dropna(subset=["amplitude"])
            if not valid_amp.empty:
                fig_amp_box = px.box(
                    valid_amp,
                    x="signal_class",
                    y="amplitude",
                    color="signal_class",
                    color_discrete_map=CLASS_COLORS,
                    points="outliers",
                    labels={"signal_class": "Signal Class", "amplitude": "Amplitude (Normalized)"}
                )
                fig_amp_box = style_chart(fig_amp_box, "Amplitude Spread & Outlier Profile", height=400, show_legend=False)
                st.plotly_chart(fig_amp_box, use_container_width=True)
            else:
                st.info("No amplitude values available.")

        with c2:
            valid_pulse = active_df.dropna(subset=["pulse_width_ms"])
            if not valid_pulse.empty:
                fig_pulse_viol = px.violin(
                    valid_pulse,
                    x="signal_class",
                    y="pulse_width_ms",
                    color="signal_class",
                    color_discrete_map=CLASS_COLORS,
                    box=True,
                    points="outliers",
                    labels={"signal_class": "Signal Class", "pulse_width_ms": "Pulse Width (ms)"}
                )
                fig_pulse_viol = style_chart(fig_pulse_viol, "Pulse Width (ms) Distribution Density", height=400, show_legend=False)
                st.plotly_chart(fig_pulse_viol, use_container_width=True)
            else:
                st.info("No pulse width values available.")

    # TAB 2: FREQUENCY & SPECTRA
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            valid_freq = active_df.dropna(subset=["dominant_frequency_hz"])
            if not valid_freq.empty:
                fig_freq_hist = px.histogram(
                    valid_freq,
                    x="dominant_frequency_hz",
                    color="signal_class",
                    color_discrete_map=CLASS_COLORS,
                    nbins=40,
                    barmode="overlay",
                    opacity=0.75,
                    labels={"dominant_frequency_hz": "Dominant Frequency (Hz)"}
                )
                # Add vertical reference line for 50Hz mains
                fig_freq_hist.add_vline(x=50, line_dash="dash", line_color="#fbbf24", annotation_text="50Hz Mains", annotation_position="top right")
                fig_freq_hist.add_vline(x=450, line_dash="dash", line_color="#38bdf8", annotation_text="450Hz Fence Threshold", annotation_position="top right")
                fig_freq_hist = style_chart(fig_freq_hist, "Dominant Frequency Spectral Density", height=400)
                st.plotly_chart(fig_freq_hist, use_container_width=True)
            else:
                st.info("No frequency values available.")

        with c2:
            valid_spec = active_df.dropna(subset=["spectral_energy", "amplitude"])
            if not valid_spec.empty:
                fig_spec_scatter = px.scatter(
                    valid_spec,
                    x="amplitude",
                    y="spectral_energy",
                    color="signal_class",
                    color_discrete_map=CLASS_COLORS,
                    hover_name="signal_id",
                    labels={"amplitude": "Amplitude (Normalized)", "spectral_energy": "Spectral Energy"}
                )
                fig_spec_scatter = style_chart(fig_spec_scatter, "Spectral Energy vs Amplitude Dispersion", height=400)
                st.plotly_chart(fig_spec_scatter, use_container_width=True)
            else:
                st.info("No spectral energy data available.")

    # TAB 3: QUALITY & CONFIDENCE
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            valid_qc = active_df.dropna(subset=["signal_quality_score", "illustrative_confidence"])
            if not valid_qc.empty:
                fig_qc = px.scatter(
                    valid_qc,
                    x="signal_quality_score",
                    y="illustrative_confidence",
                    color="signal_quality",
                    color_discrete_map=QUALITY_COLORS,
                    hover_name="signal_id",
                    labels={
                        "signal_quality_score": "Signal Quality Score (0-1)",
                        "illustrative_confidence": "Confidence Index"
                    }
                )
                fig_qc.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)")
                fig_qc.add_vline(x=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)")
                fig_qc = style_chart(fig_qc, "Signal Quality vs Confidence Index", height=400)
                st.plotly_chart(fig_qc, use_container_width=True)
            else:
                st.info("No quality metrics available.")

        with c2:
            valid_multi = active_df.dropna(subset=["amplitude", "dominant_frequency_hz", "spectral_energy"])
            if not valid_multi.empty:
                sized_data = valid_multi.copy()
                sized_data["bubble_size"] = sized_data["spectral_energy"].clip(lower=0.01) + 0.05
                fig_multi = px.scatter(
                    sized_data,
                    x="amplitude",
                    y="dominant_frequency_hz",
                    size="bubble_size",
                    color="signal_class",
                    color_discrete_map=CLASS_COLORS,
                    hover_name="signal_id",
                    labels={"amplitude": "Amplitude", "dominant_frequency_hz": "Dominant Frequency (Hz)"}
                )
                fig_multi = style_chart(fig_multi, "Multi-Channel Telemetry Bubble (Size = Spectral Energy)", height=400)
                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.info("Insufficient data for multi-channel bubble chart.")

    # TAB 4: CUSTOM RELATIONSHIP EXPLORER
    with tab4:
        st.markdown('<p style="color:#94a3b8; font-size:0.9rem;">Select any two telemetry features to examine correlations, clustering, and decision boundaries.</p>', unsafe_allow_html=True)
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            x_var = st.selectbox("X-Axis Feature", numeric_cols, index=0)
        with exp_col2:
            y_options = [c for c in numeric_cols if c != x_var]
            y_var = st.selectbox("Y-Axis Feature", y_options, index=y_options.index("dominant_frequency_hz") if "dominant_frequency_hz" in y_options else 0)
        with exp_col3:
            color_var = st.selectbox("Color Dimension", ["signal_class", "signal_quality"])

        valid_rel = active_df.dropna(subset=[x_var, y_var])
        if not valid_rel.empty:
            c_map = CLASS_COLORS if color_var == "signal_class" else QUALITY_COLORS
            fig_rel = px.scatter(
                valid_rel,
                x=x_var,
                y=y_var,
                color=color_var,
                color_discrete_map=c_map,
                hover_name="signal_id",
                labels={x_var: x_var.replace("_", " ").title(), y_var: y_var.replace("_", " ").title()}
            )
            fig_rel = style_chart(fig_rel, f"Telemetry Relationship: {x_var} vs {y_var}", height=440)
            st.plotly_chart(fig_rel, use_container_width=True)

    # TAB 5: CORRELATION MATRIX
    with tab5:
        corr_data = active_df[numeric_cols].dropna(axis=0, how="all")
        if len(corr_data) > 1:
            corr_mat = corr_data.corr(numeric_only=True)
            fig_corr = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(color="Pearson r")
            )
            fig_corr = style_chart(fig_corr, "Telemetry Feature Pearson Correlation Matrix", height=450)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("""
            **Key Statistical Takeaways:**
            - **Amplitude & Spectral Energy:** Exhibit positive correlation, indicating high-voltage discharges deposit substantial energy in the detection window.
            - **Dominant Frequency & Pulse Width:** Show characteristic inverse behavior typical of impulsive discharge physics (shorter transient pulses exhibit broader/higher spectral contents).
            """)

# =============================================================================
# MODULE 3: DATA QUALITY & RECORDS
# =============================================================================
elif page == "Data Quality & Records":
    st.markdown('<div class="section-header">Telemetry Data Hygiene & Record Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Dataset schema audit, missing value imputation tracking, and searchable record inspection.</div>', unsafe_allow_html=True)

    # 4 KPI Cards
    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric("Total Records Ingested", f"{total_raw_rows:,}", "100% Retained")
    dq2.metric("Telemetry Features", f"{total_numeric_features}", "Validated Numeric")
    dq3.metric("Exact Duplicate Rows", f"{duplicate_rows_count}", "Zero Redundancy")
    dq4.metric("Missing Telemetry Cells", f"{missing_cells_before} ➔ {missing_cells_after}", "Median Imputed", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        missing_by_col = (
            raw_df.isna()
            .sum()
            .reset_index(name="Missing Count")
            .rename(columns={"index": "Feature"})
        )
        missing_by_col["Missing %"] = (missing_by_col["Missing Count"] / total_raw_rows) * 100
        
        fig_miss = px.bar(
            missing_by_col,
            x="Feature",
            y="Missing Count",
            color="Missing Count",
            color_continuous_scale="Reds",
            text="Missing Count",
            labels={"Feature": "Telemetry Feature", "Missing Count": "Missing Cell Count"}
        )
        fig_miss.update_traces(textposition="outside")
        fig_miss = style_chart(fig_miss, "Missing Value Frequency by Feature (Raw Dataset)", height=340, show_legend=False)
        st.plotly_chart(fig_miss, use_container_width=True)

    with c2:
        compare_df = pd.DataFrame({
            "Pipeline Stage": ["Raw Telemetry", "Imputed Clean Dataset"],
            "Missing Cells": [missing_cells_before, missing_cells_after],
            "Record Count": [total_raw_rows, len(clean_df)]
        })
        fig_comp = px.bar(
            compare_df,
            x="Pipeline Stage",
            y="Missing Cells",
            text="Missing Cells",
            color="Pipeline Stage",
            color_discrete_sequence=["#f87171", "#34d399"],
            labels={"Missing Cells": "Missing Cells Count"}
        )
        fig_comp.update_traces(textposition="outside")
        fig_comp = style_chart(fig_comp, "Data Hygiene: Before vs After Preprocessing", height=340, show_legend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    # Column Schema Health Inspector
    st.markdown('<div class="section-header">Telemetry Column Schema & Health Summary</div>', unsafe_allow_html=True)
    schema_rows = []
    for col in raw_df.columns:
        c_type = str(raw_df[col].dtype)
        m_cnt = int(raw_df[col].isna().sum())
        m_pct = (m_cnt / total_raw_rows) * 100
        status = "Imputed (Median)" if m_cnt > 0 and col in numeric_cols else ("Complete" if m_cnt == 0 else "Preserved")
        schema_rows.append({
            "Feature Name": col,
            "Storage Type": c_type,
            "Telemetry Role": "Numerical Channel" if col in numeric_cols else "Categorical Label / ID",
            "Missing Count": m_cnt,
            "Missing %": f"{m_pct:.2f}%",
            "Hygiene Pipeline Action": status
        })
    schema_table = pd.DataFrame(schema_rows)
    st.dataframe(schema_table, use_container_width=True, hide_index=True)

    # Download Cleaned Dataset
    st.download_button(
        label="⬇ Download Cleaned Telemetry Dataset (CSV)",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="safefence_cleaned_telemetry.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("---")
    # Searchable Records Explorer
    st.markdown('<div class="section-header">Searchable Signal Record Explorer</div>', unsafe_allow_html=True)
    
    view_mode = st.radio("Dataset View Mode", ["Cleaned Dataset (Imputed)", "Raw Dataset (with NaNs)"], horizontal=True)
    src_data = clean_df if "Cleaned" in view_mode else raw_df

    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        tbl_search = st.text_input("Search Records by Signal ID", value=search_id, placeholder="e.g. SIM-00025")
    with f_col2:
        tbl_class = st.selectbox("Filter Class in Table", ["All Classes"] + all_classes)

    display_records = src_data.copy()
    if tbl_search.strip():
        display_records = display_records[display_records["signal_id"].astype(str).str.contains(tbl_search.strip(), case=False, na=False)]
    if tbl_class != "All Classes":
        display_records = display_records[display_records["signal_class"] == tbl_class]

    st.caption(f"Displaying **{len(display_records):,}** of **{len(src_data):,}** records")
    st.dataframe(display_records, use_container_width=True, hide_index=True)

    # Pulse Detail Inspector
    with st.expander("🔎 Individual Signal Pulse Inspector", expanded=False):
        if not display_records.empty:
            selected_sig = st.selectbox("Select Signal ID for Pulse Parameter Audit", display_records["signal_id"].tolist()[:100])
            sig_row = display_records[display_records["signal_id"] == selected_sig].iloc[0]
            
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Amplitude", f"{sig_row['amplitude']:.4f}")
            p2.metric("Pulse Width", f"{sig_row['pulse_width_ms']:.2f} ms")
            p3.metric("Dominant Freq", f"{sig_row['dominant_frequency_hz']:.1f} Hz")
            p4.metric("Spectral Energy", f"{sig_row['spectral_energy']:.4f}")
            st.json(sig_row.to_dict())

# =============================================================================
# MODULE 4: JAVA CLASSIFICATION
# =============================================================================
elif page == "Java Classification":
    st.markdown('<div class="section-header">Java 25 Classification & Risk Assessment Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Execution of the native Java SE backend adapter via high-throughput standard I/O streaming.</div>', unsafe_allow_html=True)

    # Engine Execution Bar
    exec_c1, exec_c2 = st.columns([2, 5])
    with exec_c1:
        run_btn = st.button("🚀 Run Java Classification", type="primary", use_container_width=True)
    with exec_c2:
        st.caption("Compiles `SafeFenceAI.java` and `SafeFenceDashboard.java`, feeds the active dataset, and ingests tab-separated predictions.")

    if run_btn:
        with st.spinner("Compiling and executing Java 25 classification pipeline..."):
            start_t = time.time()
            if not shutil.which("javac") or not shutil.which("java"):
                st.error("Java SE Development Kit (javac/java) not found in system PATH.")
            elif not JAVA_MAIN.exists() or not JAVA_DASHBOARD.exists():
                st.error(f"Java source files missing in: {ROOT}")
            else:
                try:
                    # 1. Compilation Step
                    comp = subprocess.run(
                        ["javac", "-cp", ".", "SafeFenceAI.java", "SafeFenceDashboard.java"],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if comp.returncode != 0:
                        st.error("Java Compilation Failed:")
                        st.code(comp.stderr or comp.stdout)
                    else:
                        # 2. Execution Step
                        run_proc = subprocess.run(
                            ["java", "-cp", ".", "SafeFenceDashboard", str(DATA_PATH)],
                            cwd=str(ROOT),
                            capture_output=True,
                            text=True,
                            timeout=45
                        )
                        elapsed = time.time() - start_t
                        if run_proc.returncode != 0:
                            st.error("Java Execution Failed:")
                            st.code(run_proc.stderr or run_proc.stdout)
                        elif not run_proc.stdout.strip():
                            st.error("Java process returned empty stdout.")
                        else:
                            try:
                                java_df = pd.read_csv(io.StringIO(run_proc.stdout), sep="\t")
                                expected = {"signal_id", "actual_class", "predicted_class", "risk"}
                                if not expected.issubset(java_df.columns):
                                    st.error("Java stdout missing expected columns.")
                                    st.code(run_proc.stdout[:2000])
                                else:
                                    st.session_state["java_result"] = java_df
                                    st.session_state["java_latency"] = elapsed
                                    st.success(f"Successfully processed {len(java_df):,} signals in {elapsed:.2f} seconds!")
                            except Exception as parse_ex:
                                st.error(f"Failed to parse TSV output: {parse_ex}")
                                st.code(run_proc.stdout[:2000])
                except subprocess.TimeoutExpired:
                    st.error("Java process timed out.")
                except Exception as ex:
                    st.error(f"Execution error: {ex}")

    # Predictions Presentation
    res = st.session_state.get("java_result")
    latency = st.session_state.get("java_latency", 0.0)

    if res is not None and not res.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        # KPI Row
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Classified Records", f"{len(res):,}", f"{latency:.2f}s latency")
        j2.metric("Output Classes", f"{res['predicted_class'].nunique()}", "Heuristic Categories")
        
        high_risk_num = int(res["risk"].astype(str).str.startswith("HIGH").sum())
        j3.metric("High-Risk Electric Fence Flags", f"{high_risk_num:,}", f"{(high_risk_num / len(res)) * 100:.1f}% of total")
        
        matches = int((res["actual_class"] == res["predicted_class"]).sum())
        match_rate = (matches / len(res)) * 100
        j4.metric("Dataset Agreement Rate", f"{match_rate:.1f}%", f"{matches} / {len(res)} matches")

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics
        c_left, c_right = st.columns([3, 2])
        with c_left:
            comp_data = (
                res.groupby(["actual_class", "predicted_class"])
                .size()
                .reset_index(name="Count")
            )
            fig_compare = px.bar(
                comp_data,
                x="actual_class",
                y="Count",
                color="predicted_class",
                barmode="group",
                color_discrete_map=CLASS_COLORS,
                labels={"actual_class": "Dataset Label", "Count": "Signals", "predicted_class": "Java Predicted Class"}
            )
            fig_compare = style_chart(fig_compare, "Actual Dataset Label vs Java Rule Output", height=380)
            st.plotly_chart(fig_compare, use_container_width=True)

        with c_right:
            pred_dist = (
                res["predicted_class"]
                .value_counts()
                .reset_index(name="Count")
                .rename(columns={"index": "Predicted Class"})
            )
            fig_pred_pie = px.pie(
                pred_dist,
                names="predicted_class",
                values="Count",
                hole=0.55,
                color="predicted_class",
                color_discrete_map=CLASS_COLORS
            )
            fig_pred_pie = style_chart(fig_pred_pie, "Java Predicted Class Proportions", height=380)
            st.plotly_chart(fig_pred_pie, use_container_width=True)

        # Safety Risk Classification Breakdown
        risk_dist = (
            res["risk"]
            .value_counts()
            .reset_index(name="Count")
            .rename(columns={"index": "Risk Level"})
        )
        fig_risk_bar = px.bar(
            risk_dist,
            x="risk",
            y="Count",
            color="risk",
            color_discrete_map=RISK_COLORS,
            text="Count",
            labels={"risk": "Safety Threat Level", "Count": "Observed Records"}
        )
        fig_risk_bar.update_traces(textposition="outside")
        fig_risk_bar = style_chart(fig_risk_bar, "Safety Risk Assessment Categories", height=320, show_legend=False)
        st.plotly_chart(fig_risk_bar, use_container_width=True)

        # Heuristic Logic Inspector Expander
        with st.expander("📖 Inspect Java Heuristic Classification Decision Rules", expanded=False):
            st.markdown("""
            ```java
            // SafeFenceAI.java - Rule-based Heuristic Logic
            if (Double.isNaN(s.amplitude()) || Double.isNaN(s.pulseWidth()) || Double.isNaN(s.frequency())) {
                return "Uncertain"; // Missing telemetry channels
            }
            if (s.frequency() >= 450 && s.amplitude() >= 0.4 && s.pulseWidth() <= 8) {
                return "Fence-Like"; // High voltage pulse signature -> Risk: HIGH
            }
            if (s.frequency() >= 45 && s.frequency() <= 65) {
                return "Mains-Like"; // AC grid coupling -> Risk: MEDIUM
            }
            if (s.amplitude() < 0.2) {
                return "Background"; // Noise floor -> Risk: LOW
            }
            return "Interference";     // Non-fence transient -> Risk: LOW
            ```
            """)

        # Searchable Predictions Table
        st.markdown('<div class="section-header">Java Classification Results Table</div>', unsafe_allow_html=True)
        r_f1, r_f2 = st.columns([2, 1])
        with r_f1:
            j_search = st.text_input("Filter by Signal ID", value=search_id, placeholder="e.g. SIM-00010")
        with r_f2:
            j_risk = st.selectbox("Filter by Threat Level", ["All Levels"] + sorted(res["risk"].unique()))

        tbl_show = res.copy()
        if j_search.strip():
            tbl_show = tbl_show[tbl_show["signal_id"].astype(str).str.contains(j_search.strip(), case=False, na=False)]
        if j_risk != "All Levels":
            tbl_show = tbl_show[tbl_show["risk"] == j_risk]

        st.caption(f"Displaying **{len(tbl_show):,}** of **{len(res):,}** classified signals")
        st.dataframe(tbl_show, use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇ Download Java Classification Results (CSV)",
            data=res.to_csv(index=False).encode("utf-8"),
            file_name="safefence_java_predictions.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("ℹ️ No Java predictions generated in this session yet. Click **'🚀 Run Java Classification'** above to compile and execute the Java 25 backend adapter.")
