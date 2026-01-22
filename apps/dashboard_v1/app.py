import streamlit as st
 
import pandas as pd
 
import numpy as np
 
import plotly.graph_objects as go
 
# 1. Page Configuration
 
st.set_page_config(page_title="Ohmium Dashboard v1", layout="wide")
 
st.title("Ohmium Stack Monitoring - Dashboard v1")
 
st.markdown("Monitor and analyze physical parameters alongside health scores for root cause diagnostics.")
 
# 2. File Uploader
 
st.sidebar.header("1. Data Ingestion")
 
uploaded_files = st.sidebar.file_uploader(
 
    "Upload up to 3 BST CSV log files (24h)",
 
    type=["csv"],
 
    accept_multiple_files=True
 
)
 
if not uploaded_files:
 
    st.info("Please upload BST logfiles in the sidebar to proceed.")
 
    st.stop()
 
# 3. Data Loading
 
@st.cache_data
 
def load_and_standardize(file):
 
    try:
 
        df = pd.read_csv(file, low_memory=False)
 
        df.columns = df.columns.str.strip()
 
        if "Time" in df.columns:
 
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
 
        else:
 
            return None
 
        return df.dropna(subset=["Time"]).sort_values("Time")
 
    except Exception:
 
        return None
 
# 4. Health Engine (Voltage: 40, Cell Spread: 30, Flow: 20, Cond: 10)
 
WEIGHTS = {"voltage": 40, "spread": 30, "flow": 20, "cond": 10}
 
RED_THRESHOLD = 60
 
def process_stack_health(df):
 
    df = df.copy()
 
    cell_cols = [c for c in df.columns if c.startswith("Cell")]
 
    if cell_cols:
 
        df[cell_cols] = df[cell_cols].apply(pd.to_numeric, errors="coerce")
 
        df["avg_cell_voltage"] = df[cell_cols].mean(axis=1)
 
        df["cell_spread"] = df[cell_cols].max(axis=1) - df[cell_cols].min(axis=1)
 
    psu_i_cols = [c for c in df.columns if "OP I" in c]
 
    if psu_i_cols:
 
        df["stack_current"] = df[psu_i_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
 
    base_v = df["avg_cell_voltage"].median()
 
    base_s = df["cell_spread"].median()
 
    base_f = pd.to_numeric(df.get("FLM 101", 1), errors='coerce').median()
 
    base_c = pd.to_numeric(df.get("COS 101", 1), errors='coerce').median()
 
    df["p_vol"] = ((df["avg_cell_voltage"] - base_v) / base_v).clip(0)
 
    df["p_spr"] = ((df["cell_spread"] - base_s) / base_s).clip(0)
 
    df["p_flo"] = ((base_f - pd.to_numeric(df.get("FLM 101", 0), errors='coerce')) / base_f).clip(0)
 
    df["p_con"] = ((pd.to_numeric(df.get("COS 101", 0), errors='coerce') - base_c) / base_c).clip(0)
 
    df["health_score"] = (100 - (
 
        df["p_vol"] * WEIGHTS["voltage"] +
 
        df["p_spr"] * WEIGHTS["spread"] +
 
        df["p_flo"] * WEIGHTS["flow"] +
 
        df["p_con"] * WEIGHTS["cond"]
 
    )).clip(0, 100)
 
    return df
 
# 5. Process Data
 
dfs = {}
 
for file in uploaded_files[:3]:
 
    data = load_and_standardize(file)
 
    if data is not None:
 
        dfs[file.name] = process_stack_health(data)
 
# --- SECTION 1: STACK RANKING ---
 
st.header("1. Stack Health Ranking")
 
ranking_data = [{"Stack": n, "Avg Health": round(df["health_score"].mean(), 2)} for n, df in dfs.items()]
 
st.table(pd.DataFrame(ranking_data).sort_values("Avg Health", ascending=False))
 
# --- SECTION 2: CHARTS ---
 
st.header("2. Performance Dashboards")
 
selected_stacks = st.multiselect("Select Stacks:", list(dfs.keys()), default=list(dfs.keys())[:1])
 
# Physical Parameters
 
PARAM_MAP = {"Avg Cell Voltage": "avg_cell_voltage", "Flow Rate (FLM 101)": "FLM 101", "Pressure (PRT 102)": "PRT 102", "Conductivity (COS 101)": "COS 101", "Stack Current": "stack_current"}
 
sel_p = st.selectbox("Select Parameter:", list(PARAM_MAP.keys()))
 
st.subheader("Physical Parameter Comparison:")
 
if selected_stacks:
 
    fig_p = go.Figure()
 
    for n in selected_stacks:
 
        fig_p.add_trace(go.Scatter(x=dfs[n]["Time"], y=dfs[n][PARAM_MAP[sel_p]], name=n, mode='lines'))
 
    st.plotly_chart(fig_p, use_container_width=True)
 
# Health Score
st.subheader("Health Score Over Time:")
 
if selected_stacks:
 
    fig_h = go.Figure()
 
    fig_h.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.1, annotation_text="GREEN")
 
    fig_h.add_hrect(y0=60, y1=80, fillcolor="orange", opacity=0.1, annotation_text="AMBER")
 
    fig_h.add_hrect(y0=0, y1=60, fillcolor="red", opacity=0.1, annotation_text="RED")
 
    for n in selected_stacks:
 
        fig_h.add_trace(go.Scatter(x=dfs[n]["Time"], y=dfs[n]["health_score"], name=n, mode='lines'))
 
    st.plotly_chart(fig_h, use_container_width=True)
 
# --- SECTION 3: DIAGNOSTICS ---
 
st.header("3. Detailed Diagnostics")
 
target = st.selectbox("Deep-Dive Stack:", selected_stacks)
 
if target:
 
    df = dfs[target]
 
    # Zone Distribution
 
    pts = len(df)
 
    c1, c2, c3 = st.columns(3)
 
    c1.metric("Green %", f"{(df['health_score']>=80).sum()/pts*100:.1f}%")
 
    c2.metric("Amber %", f"{((df['health_score']>=60)&(df['health_score']<80)).sum()/pts*100:.1f}%")
 
    c3.metric("Red %", f"{(df['health_score']<60).sum()/pts*100:.1f}%")
 
    # Root Cause Inspection
 
    st.subheader("Root Cause Inspection")
 
    time_opts = df["Time"].dt.strftime('%H:%M:%S').tolist()
 
    sel_time = st.select_slider("Select Timestamp:", options=time_opts)
 
    row = df[df["Time"].dt.strftime('%H:%M:%S') == sel_time].iloc[0]
 
    penalties = {
 
        "increased cell voltage": row["p_vol"] * WEIGHTS["voltage"],
 
        "high cell voltage spread": row["p_spr"] * WEIGHTS["spread"],
 
        "low water flow": row["p_flo"] * WEIGHTS["flow"],
 
        "rising water conductivity": row["p_con"] * WEIGHTS["cond"]
 
    }
 
    # Sort and filter active drivers
 
    active_drivers = {k: v for k, v in penalties.items() if v > 0}
 
    sorted_drivers = sorted(active_drivers.items(), key=lambda x: x[1], reverse=True)
 
    # --- AUTOMATED SUMMARY SENTENCE ---
 
    if not sorted_drivers:
 
        summary = "Stack is operating at optimal baseline levels with no significant penalties."
 
    else:
 
        top_cause = sorted_drivers[0][0]
 
        summary = f"Health score reduction at {sel_time} is primarily driven by **{top_cause}**"
 
        if len(sorted_drivers) > 1:
 
            summary += f", with secondary contribution from **{sorted_drivers[1][0]}**."
 
        else:
 
            summary += "."
 
    st.markdown(f"> {summary}")
 
    st.table(pd.DataFrame.from_dict(penalties, orient='index', columns=['Points Subtracted']).sort_values("Points Subtracted", ascending=False))
 