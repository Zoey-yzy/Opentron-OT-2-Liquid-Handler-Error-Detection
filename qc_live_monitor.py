"""
OT-2 Real-Time Dashboard
Reads live_state.json (written by pipeline.py) and pipeline_log.csv.
Run: streamlit run dashboard.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LIVE_STATE_FILE = Path("live_state.json")
LOG_FILE = Path("pipeline_log.csv")
REFRESH_MS = 2000

st.set_page_config(page_title="OT-2 Mission Control", layout="wide", page_icon="🛰️")

# --- Auto-refresh every REFRESH_MS milliseconds ---
# pip install streamlit-autorefresh  (one-time install)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_MS, key="pipeline_refresh")
except ImportError:
    st.warning("Install streamlit-autorefresh for live updates: `pip install streamlit-autorefresh`")
    if st.button("Manual Refresh"):
        st.rerun()

# --- Load live state ---
def load_live_state() -> dict:
    if not LIVE_STATE_FILE.exists():
        return {"status": "idle"}
    try:
        with open(LIVE_STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"status": "idle"}

# --- Load CSV log ---
def load_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame()

state = load_live_state()
log_df = load_log()

status = state.get("status", "idle")
current_well = state.get("current_well", "—")
expected_vol = state.get("expected_vol", 0.0)
detected_vol = state.get("detected_vol")
deviation_pct = state.get("deviation_pct")
error_flag = state.get("error_flag")
wells_completed = state.get("wells_completed", 0)
wells_total = state.get("wells_total", 96)
annotated_image = state.get("annotated_image")
mpi = state.get("mpi", 0)
camera_connected = state.get("camera_connected", False)
pipette_w_px = state.get("pipette_w_px", 0.0)
pipette_y = state.get("pipette_y", 0.0)
liquid_y = state.get("liquid_y", 0.0)
z_height = state.get("z_height", 0.0)

# --- Title row ---
status_colors = {
    "idle":     ("⚪", "gray"),
    "running":  ("🟡", "orange"),
    "complete": ("🟢", "green"),
    "error":    ("🔴", "red"),
    "stopped":  ("🔴", "red"),
}
icon, _ = status_colors.get(status, ("⚪", "gray"))

st.title(f"🛰️ OT-2 Mission Control  {icon} {status.upper()}")

# --- Sidebar ---
st.sidebar.header("👤 Human-in-the-Loop")
st.sidebar.write("Confirm physical states before each run:")
check_pos      = st.sidebar.checkbox("Camera Alignment", value=False)
check_contact  = st.sidebar.checkbox("Tip-to-Liquid Contact", value=False)
check_labware  = st.sidebar.checkbox("Correct Tiprack Loaded", value=False)
check_safety   = st.sidebar.checkbox("Workspace Clear", value=False)
human_qc_pass  = check_pos and check_contact and check_labware and check_safety

st.sidebar.divider()
st.sidebar.metric("Wells Completed", f"{wells_completed} / {wells_total}")
if wells_total > 0:
    st.sidebar.progress(wells_completed / wells_total)

if st.sidebar.button("🚨 EMERGENCY HALT", type="primary"):
    st.session_state["halt"] = True
    st.sidebar.error("HALT REQUESTED — stop pipeline.py manually.")

# --- Row 1: Status cards ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    if camera_connected:
        st.success("📷 Camera: ONLINE")
    else:
        st.error("📷 Camera: OFFLINE")

with c2:
    st.metric("Current Well", current_well)

with c3:
    if detected_vol is not None:
        delta = detected_vol - expected_vol
        st.metric(
            "Detected Volume",
            f"{detected_vol:.1f} µL",
            delta=f"{delta:+.1f} µL",
            delta_color="normal" if abs(delta) <= expected_vol * 0.15 else "inverse",
        )
    else:
        st.metric("Detected Volume", "—")

with c4:
    if deviation_pct is not None:
        st.metric("Deviation", f"{deviation_pct:.1f}%",
                  delta_color="normal" if deviation_pct < 15 else "inverse")
    else:
        st.metric("Deviation", "—")

st.divider()

# --- Row 2: QC diagnostics + verdict ---
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("🛡️ Automated Hardware & Vision QC")

    # Camera
    if camera_connected:
        st.write("✅ **Camera:** Connected")
    else:
        st.write("❌ **Camera:** Disconnected")

    # Image intensity
    if mpi == 0:
        st.write("⚪ **Image Intensity:** No data yet")
    elif 40 <= mpi <= 220:
        st.write(f"✅ **Image Intensity:** {mpi} MPI (range: 40–220)")
    else:
        st.write(f"❌ **Image Intensity:** {mpi} MPI — OUT OF RANGE")

    # Coordinate sanity
    if liquid_y == 0 and pipette_y == 0:
        st.write("⚪ **Coordinate Logic:** No data yet")
    elif liquid_y > pipette_y:
        st.write(f"✅ **Coordinate Logic:** Liquid below tip (+{liquid_y - pipette_y:.1f} px)")
    else:
        st.write(f"❌ **Coordinate Logic:** Liquid ABOVE tip (invalid geometry)")

    # Tip width integrity
    if pipette_w_px == 0:
        st.write("⚪ **Tip Integrity:** No data yet")
    elif abs(pipette_w_px - 52.0) < 3.0:
        st.write(f"✅ **Tip Width:** {pipette_w_px:.1f} px (±3 px of nominal)")
    else:
        st.write(f"❌ **Tip Width:** {pipette_w_px:.1f} px — possible bent tip/crash")

    # Volume tolerance
    if detected_vol is not None and expected_vol > 0:
        tol = state.get("tolerance_pct", 15.0)
        if deviation_pct <= tol:
            st.write(f"✅ **Volume:** {detected_vol:.1f} µL (Δ {deviation_pct:.1f}% ≤ {tol}%)")
        else:
            st.write(f"❌ **Volume:** {detected_vol:.1f} µL (Δ {deviation_pct:.1f}% > {tol}% tolerance)")
    else:
        st.write("⚪ **Volume:** No data yet")

with col_right:
    st.subheader("⚖️ Operational Verdict")

    auto_qc_pass = (
        camera_connected
        and (40 <= mpi <= 220 if mpi else False)
        and (liquid_y > pipette_y if liquid_y or pipette_y else False)
        and (abs(pipette_w_px - 52.0) < 3.0 if pipette_w_px else False)
        and (error_flag == 0 if error_flag is not None else True)
    )

    if status == "idle":
        st.info("### ⚪ STATUS: WAITING\nStart pipeline.py to begin.")
    elif status == "complete":
        st.success("### 🟢 PROTOCOL COMPLETE\nAll wells processed successfully.")
    elif status in ("error", "stopped"):
        st.error(f"### 🔴 HALTED at well {current_well}")
        if error_flag:
            st.warning("Volume error exceeded tolerance — liquid returned to reservoir.")
    elif auto_qc_pass and human_qc_pass:
        st.success("### 🟢 STATUS: NOMINAL\nAll QC checks passing.")
    else:
        st.error("### 🔴 STATUS: QC FAIL")
        if not auto_qc_pass:
            st.warning("Automated metric out of range.")
        if not human_qc_pass:
            st.info("Pending operator verification (sidebar checkboxes).")

st.divider()

# --- Row 3: Gauge + annotated image ---
bottom_l, bottom_m, bottom_r = st.columns([1, 1, 1])

with bottom_l:
    gauge_val = detected_vol if detected_vol is not None else expected_vol
    gauge_min = max(0, expected_vol * 0.7)
    gauge_max = expected_vol * 1.3
    green_lo  = expected_vol * 0.85
    green_hi  = expected_vol * 1.15

    bar_color = "red" if (error_flag == 1) else "#00cc96"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        delta={"reference": expected_vol, "valueformat": ".1f"},
        title={"text": "Detected Volume (µL)"},
        gauge={
            "axis": {"range": [gauge_min, gauge_max]},
            "bar":  {"color": bar_color},
            "steps": [{"range": [green_lo, green_hi], "color": "#e0ffe0"}],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": expected_vol,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with bottom_m:
    st.write("**Model Context**")
    mae = state.get("mae", "—")
    r2  = state.get("r2", "—")
    n   = state.get("sample_count", "—")
    st.code(f"""
MAE:          {mae} µL
R²:           {r2}
Samples:      {n}
Tolerance:    ±{state.get('tolerance_pct', 15.0):.0f}%
    """)

with bottom_r:
    if annotated_image and Path(annotated_image).exists():
        st.image(annotated_image, caption=f"Well {current_well} — annotated", use_container_width=True)
    else:
        st.info("No image yet — waiting for first detection.")

# --- Row 4: Trend chart ---
if not log_df.empty and "volume_ul" in log_df.columns and "expected_vol" in log_df.columns:
    st.divider()
    st.subheader("📈 Volume Trend")
    plot_df = log_df.dropna(subset=["volume_ul"]).tail(50)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=plot_df["well"], y=plot_df["expected_vol"],
        mode="lines", name="Expected", line=dict(color="gray", dash="dash"),
    ))
    fig2.add_trace(go.Scatter(
        x=plot_df["well"], y=plot_df["volume_ul"],
        mode="lines+markers", name="Detected",
        marker=dict(
            color=["red" if e else "#00cc96" for e in plot_df["error_flag"].fillna(0)],
            size=8,
        ),
        line=dict(color="#636efa"),
    ))
    fig2.update_layout(height=260, margin=dict(t=10, b=0),
                       xaxis_title="Well", yaxis_title="µL")
    st.plotly_chart(fig2, use_container_width=True)

# --- Row 5: Audit log ---
st.divider()
st.subheader("📋 Session Audit Log")
if log_df.empty:
    st.info("No entries yet.")
else:
    display_cols = ["timestamp", "well", "expected_vol", "volume_ul", "error_flag", "decision"]
    available = [c for c in display_cols if c in log_df.columns]
    styled = log_df[available].tail(20).copy()
    if "error_flag" in styled.columns:
        styled["error_flag"] = styled["error_flag"].map({0: "✅ OK", 1: "❌ ERROR"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption(f"Last refreshed: {time.strftime('%H:%M:%S')}  |  State file: {LIVE_STATE_FILE}")
