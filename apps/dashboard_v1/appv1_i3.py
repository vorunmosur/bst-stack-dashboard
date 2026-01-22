import streamlit as st
 
import pandas as pd
 
import numpy as np
 
import plotly.graph_objects as go
 
# 1. Page Configuration
 
st.set_page_config(page_title="Ohmium Dashboard v1", layout="wide")
 
st.title("Ohmium Stack Monitoring - Dashboard v1")
 
st.markdown("Interactive analysis of physical parameters and sustained health alerts.")
 
# 2. File Uploader
 
st.sidebar.header("1. Upload BST Log Files")
 
uploaded_files = st.sidebar.file_uploader(
 
    "Upload up to 3 BST CSV log files",
 
    type=["csv"],
 
    accept_multiple_files=True
 
)
 
if not uploaded_files:
 
    st.info("Please upload BST logfiles in the sidebar to proceed.")
 
    st.stop()
 
# 3. Data Loading Logic
 
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
 
# 4. Health & Penalty Logic (Voltage 40, Spread 30)
 
WEIGHTS = {"voltage": 40, "spread": 30, "flow": 20, "cond": 10}
 
RED_THRESHOLD = 60
 
AMBER_THRESHOLD = 80
 
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
 
# 5. Process Files
 
dfs = {}
 
for file in uploaded_files[:3]:
 
    data = load_and_standardize(file)
 
    if data is not None:
 
        dfs[file.name] = process_stack_health(data)
 
st.header("1. Performance Dashboards")
 
selected_stacks = st.multiselect("Choose stacks to compare:", list(dfs.keys()), default=list(dfs.keys())[:1])
 
# --- Plot A: Physical Parameters ---
 
st.subheader("Physical Parameter Comparison")
 
PARAM_MAP = {
 
    "Avg Cell Voltage": "avg_cell_voltage",
 
    "Flow Rate (FLM 101)": "FLM 101",
 
    "Pressure (PRT 102)": "PRT 102",
 
    "Conductivity (COS 101)": "COS 101",
 
    "Stack Current": "stack_current"
 
}
 
selected_param = st.selectbox("Select Parameter to Plot:", list(PARAM_MAP.keys()))
 
if selected_stacks:
 
    fig_params = go.Figure()
 
    for name in selected_stacks:
 
        p_col = PARAM_MAP[selected_param]
 
        if p_col in dfs[name].columns:
 
            fig_params.add_trace(go.Scatter(
 
                x=dfs[name]["Time"], y=dfs[name][p_col], name=name, mode='lines',
 
                hovertemplate="<b>Stack:</b> %{fullData.name}<br><b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>"
 
            ))
 
    fig_params.update_layout(height=400, hovermode="x unified")
 
    st.plotly_chart(fig_params, use_container_width=True)
 
# --- Plot B: Health Score ---
 
st.subheader("Stack Health Score (0-100)")
 
if selected_stacks:
 
    fig_health = go.Figure()
 
    fig_health.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.1, annotation_text="GREEN")
 
    fig_health.add_hrect(y0=60, y1=80, fillcolor="orange", opacity=0.1, annotation_text="AMBER")
 
    fig_health.add_hrect(y0=0, y1=60, fillcolor="red", opacity=0.1, annotation_text="RED")
 
    for name in selected_stacks:
 
        h_scores = dfs[name]["health_score"]
 
        zones = ["RED" if s < 60 else "AMBER" if s < 80 else "GREEN" for s in h_scores]
 
        fig_health.add_trace(go.Scatter(
 
            x=dfs[name]["Time"], y=h_scores, name=name, mode='lines', customdata=zones,
 
            hovertemplate="<b>Stack:</b> %{fullData.name}<br><b>Time:</b> %{x}<br><b>Health:</b> %{y:.1f}<br><b>Zone:</b> %{customdata}<extra></extra>"
 
        ))
 
    fig_health.update_layout(height=400, hovermode="x unified", yaxis_range=[0, 105])
 
    st.plotly_chart(fig_health, use_container_width=True)
 
# 6. Sustained Alert Logic & Root Cause
 
st.header("2. Detailed Root Cause & Alerts")
 
target = st.selectbox("Analyze stack details:", selected_stacks)
 
if target:
 
    df = dfs[target]
 
    # Identify Sustained Red Events (Continuous Red for 30 samples)
 
    df["is_red"] = (df["health_score"] < RED_THRESHOLD).astype(int)
 
    # Grouping consecutive red points to find episodes
 
    df["group"] = (df["is_red"] != df["is_red"].shift()).cumsum()
 
    red_episodes = df[df["is_red"] == 1].groupby("group").agg(
 
        Start_Time=("Time", "min"),
 
        End_Time=("Time", "max"),
 
        Duration_Samples=("Time", "count")
 
    )
 
    # Filter for sustained events (e.g., duration > 30 minutes/samples)
 
    sustained_table = red_episodes[red_episodes["Duration_Samples"] >= 30]
 
    st.subheader("Sustained Red Alerts (30m+ Episodes)")
 
    if not sustained_table.empty:
 
        st.warning(f"Detected {len(sustained_table)} sustained critical events.")
 
        st.table(sustained_table[["Start_Time", "End_Time", "Duration_Samples"]])
 
    else:
 
        st.success("No sustained critical alerts detected in this log.")
 
    # Time Selector for Inspection
 
    st.subheader("Root Cause Inspection")
 
    time_options = df["Time"].dt.strftime('%H:%M:%S').tolist()
 
    selected_time_str = st.select_slider("Select time:", options=time_options)
 
    row = df[df["Time"].dt.strftime('%H:%M:%S') == selected_time_str].iloc[0]
 
    zone = "GREEN" if row['health_score'] >= 80 else "AMBER" if row['health_score'] >= 60 else "RED"
 
    st.info(f"At **{selected_time_str}**, Health: **{row['health_score']:.1f}** ({zone} Zone)")
 
    drivers = {
 
        "Voltage Penalty": row["p_vol"] * WEIGHTS["voltage"],
 
        "Cell Spread Penalty": row["p_spr"] * WEIGHTS["spread"],
 
        "Flow Rate Penalty": row["p_flo"] * WEIGHTS["flow"],
 
        "Conductivity Penalty": row["p_con"] * WEIGHTS["cond"]
 
    }
 
    explanation_df = pd.DataFrame.from_dict(drivers, orient='index', columns=['Points Subtracted'])
 
    st.table(explanation_df.sort_values(by="Points Subtracted", ascending=False))
 
    # Download Feature
 
    csv = explanation_df.to_csv().encode('utf-8')
 
    st.download_button("Download Root Cause Table", data=csv, file_name=f"root_cause_{selected_time_str}.csv", mime='text/csv')
 
 