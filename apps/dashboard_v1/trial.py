# final perhaps
 
import streamlit as st

import pandas as pd

import numpy as np

import plotly.graph_objects as go
 
# 1. Page Configuration & Styling

st.set_page_config(page_title="Ohmium Stack Monitor", layout="wide")
 
st.markdown("""
<style>

    .main { background-color: #f8f9fa; }

    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }

    .status-panel { 

        padding: 20px; 

        border-radius: 10px; 

        margin: 10px 0; 

        border-left: 8px solid;

        font-size: 1.1em;

        line-height: 1.5;

    }
</style>

    """, unsafe_allow_html=True)
 
st.title("Ohmium Stack Health Dashboard")

st.markdown("Monitor and analyze physical parameters alongside health score.")
 
# 2. Sidebar Data Ingestion

with st.sidebar:

    st.header("1. Data Ingestion")

    uploaded_files = st.file_uploader("Upload BST Log Files (24h CSV)", type=["csv"], accept_multiple_files=True)

    if not uploaded_files:

        st.info("Awaiting CSV logs...")

        st.stop()

    st.divider()

    st.header("2. Visual Controls")

    PARAM_MAP = {

        "Avg Cell Voltage": "avg_cell_voltage", 

        "Flow Rate (FLM 101)": "FLM 101", 

        "Pressure (PRT 102)": "PRT 102", 

        "Conductivity (COS 101)": "COS 101", 

        "Stack Current": "stack_current"

    }

    selected_param = st.selectbox("Select Physical Parameter", list(PARAM_MAP.keys()))
 
# 3. Core Processing Engine

@st.cache_data

def load_and_process(file):

    df = pd.read_csv(file, low_memory=False)

    df.columns = df.columns.str.strip()

    time_col = next((c for c in df.columns if 'time' in c.lower()), 'Time')

    df['Time'] = pd.to_datetime(df[time_col], errors='coerce')

    df = df.dropna(subset=['Time']).sort_values('Time')

    # Robust Initialization to prevent KeyErrors

    df["avg_cell_voltage"] = np.nan

    df["cell_spread"] = np.nan

    df["stack_current"] = np.nan
 
    # Enrichment: Voltages

    cell_cols = [c for c in df.columns if c.startswith("Cell")]

    if cell_cols:

        df[cell_cols] = df[cell_cols].apply(pd.to_numeric, errors='coerce')

        df["avg_cell_voltage"] = df[cell_cols].mean(axis=1)

        df["cell_spread"] = df[cell_cols].max(axis=1) - df[cell_cols].min(axis=1)

    # Enrichment: Current (Sum of PSU OP I columns)

    psu_i_cols = [c for c in df.columns if "OP I" in c]

    if psu_i_cols:

        df["stack_current"] = df[psu_i_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)

    # Self-Baseline Penalties (Phase 4 Logic)

    base = {

        "v": df["avg_cell_voltage"].median() if not df["avg_cell_voltage"].isna().all() else 1.5,

        "s": df["cell_spread"].median() if not df["cell_spread"].isna().all() else 0.05,

        "f": pd.to_numeric(df.get("FLM 101", 1), errors='coerce').median(),

        "c": pd.to_numeric(df.get("COS 101", 1), errors='coerce').median()

    }

    df["p_v"] = ((df["avg_cell_voltage"] - base["v"]) / base["v"]).clip(0).fillna(0)

    df["p_s"] = ((df["cell_spread"] - base["s"]) / base["s"]).clip(0).fillna(0)

    df["p_f"] = ((base["f"] - pd.to_numeric(df.get("FLM 101", 0), errors='coerce')) / base["f"]).clip(0).fillna(0)

    df["p_c"] = ((pd.to_numeric(df.get("COS 101", 0), errors='coerce') - base["c"]) / base["c"]).clip(0).fillna(0)

    # Weights: Voltage 40, Spread 30, Flow 20, Cond 10

    df["health_score"] = (100 - (df["p_v"]*40 + df["p_s"]*30 + df["p_f"]*20 + df["p_c"]*10)).clip(0, 100)

    return df
 
stack_data = {f.name: load_and_process(f) for f in uploaded_files[:3]}

selected_stacks = st.multiselect("Select Stacks to Compare", list(stack_data.keys()), default=list(stack_data.keys())[:1])
 
# SECTION 1: RANKING

st.subheader("📊 24h Stack Performance Ranking")

rank_list = [{"Stack": n, "Avg Health": d["health_score"].mean()} for n, d in stack_data.items()]

st.dataframe(pd.DataFrame(rank_list).sort_values("Avg Health", ascending=False), use_container_width=True)
 
# SECTION 2: TRENDS (Physical first, then Health)

st.subheader(f"Trend: {selected_param}")

fig_p = go.Figure()

for n in selected_stacks:

    p_col = PARAM_MAP[selected_param]

    if p_col in stack_data[n].columns:

        fig_p.add_trace(go.Scatter(x=stack_data[n]['Time'], y=stack_data[n][p_col], name=n, mode='lines'))

fig_p.update_layout(margin=dict(l=0,r=0,b=0,t=10), xaxis_title="Time", hovermode="x unified")

st.plotly_chart(fig_p, use_container_width=True)
 
st.subheader("Trend: Stack Health Score")

fig_h = go.Figure()

for z, color, y_range in [("RED", "red", [0,60]), ("AMBER", "orange", [60,80]), ("GREEN", "green", [80,100])]:

    fig_h.add_hrect(y0=y_range[0], y1=y_range[1], fillcolor=color, opacity=0.1, line_width=0, annotation_text=z)

for n in selected_stacks:

    fig_h.add_trace(go.Scatter(x=stack_data[n]['Time'], y=stack_data[n]['health_score'], name=n, mode='lines'))

fig_h.update_layout(margin=dict(l=0,r=0,b=0,t=10), xaxis_title="Time", yaxis_range=[0,105], hovermode="x unified")

st.plotly_chart(fig_h, use_container_width=True)
 
# SECTION 3: DIAGNOSTICS

st.divider()

st.header("🔍 Deep-Dive Diagnostics")

target = st.selectbox("Select Stack for Detailed Analysis", selected_stacks)
 
if target:

    df = stack_data[target]

    m1, m2, m3 = st.columns(3)

    m1.metric("Green Zone %", f"{(df['health_score']>=80).sum()/len(df)*100:.1f}%")

    m2.metric("Amber Zone %", f"{((df['health_score']>=60)&(df['health_score']<80)).sum()/len(df)*100:.1f}%")

    m3.metric("Red Zone %", f"{(df['health_score']<60).sum()/len(df)*100:.1f}%")
 
    st.subheader("🚨 Sustained Red Alert Log (10+ Consecutive Points)")

    df["is_red"] = (df["health_score"] < 60).astype(int)

    df["grp"] = (df["is_red"] != df["is_red"].shift()).cumsum()

    red_eps = df[df["is_red"] == 1].groupby("grp").agg(Start=("Time", "min"), End=("Time", "max"), Points=("Time", "count"))

    sustained = red_eps[red_eps["Points"] >= 10]

    if not sustained.empty:

        st.warning(f"Detected {len(sustained)} sustained red events.")

        st.table(sustained.reset_index(drop=True))

    else:

        st.success("No sustained red events (10+ points) detected.")
 
    st.subheader("Root Cause Summary & Inspection")

    time_strs = df['Time'].dt.strftime('%H:%M:%S').tolist()

    sel_time = st.select_slider("Select Timestamp to Inspect", options=time_strs)

    row = df[df['Time'].dt.strftime('%H:%M:%S') == sel_time].iloc[0]

    st.write(f"### Selected Time: {sel_time} | Health: {row['health_score']:.1f}")

    zone_name = "GREEN" if row['health_score'] >= 80 else "AMBER" if row['health_score'] >= 60 else "RED"

    bg_color = "#e8f5e9" if zone_name == "GREEN" else "#fff3e0" if zone_name == "AMBER" else "#ffebee"

    border_color = "#2e7d32" if zone_name == "GREEN" else "#ef6c00" if zone_name == "AMBER" else "#c62828"

    p_map = {"increased voltage": row["p_v"]*40, "high cell spread": row["p_s"]*30, "low water flow": row["p_f"]*20, "rising conductivity": row["p_c"]*10}

    active = sorted({k: v for k, v in p_map.items() if v > 0}.items(), key=lambda x: x[1], reverse=True)

    summary = f"Health drop is primarily driven by <b>{active[0][0]}</b>" if active else "Stack is currently operating at optimal baseline levels."

    if len(active) > 1: summary += f", with secondary contribution from <b>{active[1][0]}</b>."
 
    st.markdown(f"""
<div class="status-panel" style="background-color: {bg_color}; border-left-color: {border_color}; color: #333;">
<strong>Status:</strong> The stack is currently in the <b>{zone_name}</b> zone.<br>
<strong>Diagnosis:</strong> {summary}
</div>

    """, unsafe_allow_html=True)
 
    res_df = pd.DataFrame.from_dict(p_map, orient='index', columns=['Points Subtracted']).sort_values("Points Subtracted", ascending=False)

    st.table(res_df)
 
    csv = res_df.to_csv().encode('utf-8')

    st.download_button("Download Diagnostic Report", data=csv, file_name=f"report_{target}_{sel_time}.csv", mime='text/csv')
 