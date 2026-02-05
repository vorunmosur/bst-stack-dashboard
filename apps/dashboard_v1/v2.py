import streamlit as st

import pandas as pd

import numpy as np

import plotly.graph_objects as go

from plotly.subplots import make_subplots
 
# 1. Page Configuration & Professional Styling

st.set_page_config(page_title="Ohmium Stack Monitor", layout="wide")

st.markdown("""
<style>

    .main { background-color: #f8f9fa; }

    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }

    .status-panel { padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 8px solid; font-size: 1.1em; line-height: 1.5; }
</style>

    """, unsafe_allow_html=True)
 
st.title("⚡ Ohmium Stack Health Dashboard")

st.markdown("Monitor and analyze physical parameters alongside health scores for root cause diagnostics.")
 
# 2. Universal Data Processing Engine (BST & MTS)

@st.cache_data

def load_and_process(file, mode):

    df = pd.read_csv(file, low_memory=False)

    df.columns = df.columns.str.strip()

    # Time Parsing

    time_col = 'TimeStamp' if mode == "MTS" else next((c for c in df.columns if 'time' in c.lower()), 'Time')

    df['Time'] = pd.to_datetime(df[time_col], errors='coerce')

    df = df.dropna(subset=['Time']).sort_values('Time')

    # Column Mapping and Initialization

    if mode == "BST":

        cell_cols = [c for c in df.columns if c.startswith("Cell")]

        psu_i_cols = [c for c in df.columns if "OP I" in c]

        df["stack_current"] = df[psu_i_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1) if psu_i_cols else np.nan

        flow_col, cond_col = "FLM 101", "COS 101"

    else: # MTS

        cell_cols = [c for c in df.columns if c.startswith("cV")]

        df["stack_current"] = pd.to_numeric(df.get("psI(A)", np.nan), errors='coerce')

        flow_col, cond_col = "wF(LPM)", "wC(µS/cm)"
 
    if cell_cols:

        df[cell_cols] = df[cell_cols].apply(pd.to_numeric, errors='coerce')

        df["avg_cell_voltage"] = df[cell_cols].mean(axis=1)

        df["cell_spread"] = df[cell_cols].max(axis=1) - df[cell_cols].min(axis=1)
 
    # Health Engine Logic (Weights: 40, 30, 20, 10)

    base = {

        "v": df["avg_cell_voltage"].median() if "avg_cell_voltage" in df.columns else 1,

        "s": df["cell_spread"].median() if "cell_spread" in df.columns else 0.05,

        "f": pd.to_numeric(df.get(flow_col, 1), errors='coerce').median(),

        "c": pd.to_numeric(df.get(cond_col, 1), errors='coerce').median()

    }

    df["p_v"] = ((df["avg_cell_voltage"] - base["v"]) / base["v"]).clip(0).fillna(0)

    df["p_s"] = ((df["cell_spread"] - base["s"]) / base["s"]).clip(0).fillna(0)

    df["p_f"] = ((base["f"] - pd.to_numeric(df.get(flow_col, 0), errors='coerce')) / base["f"]).clip(0).fillna(0)

    df["p_c"] = ((pd.to_numeric(df.get(cond_col, 0), errors='coerce') - base["c"]) / base["c"]).clip(0).fillna(0)

    df["health_score"] = (100 - (df["p_v"]*40 + df["p_s"]*30 + df["p_f"]*20 + df["p_c"]*10)).clip(0, 100)

    return df
 
# 3. Sidebar

with st.sidebar:

    st.header("1. Data Ingestion")

    sys_mode = st.radio("Select System Type", ["BST", "MTS"])

    uploaded_files = st.file_uploader(f"Upload {sys_mode} CSV Files", type=["csv"], accept_multiple_files=True)
 
if not uploaded_files:

    st.info("👈 Please select the system type and upload CSV files in the sidebar.")

    st.stop()
 
# 4. Global Data Loading

all_data = {f.name: load_and_process(f, sys_mode) for f in uploaded_files}
 
# 5. UI Tabs

tab1, tab2 = st.tabs(["📋 Dashboard & Diagnostics", "📈 Comparative Trend Analysis"])
 
# --- TAB 1: DASHBOARD ---

with tab1:

    selected_stacks = st.multiselect("Select Stacks:", list(all_data.keys()), default=list(all_data.keys())[:1])

    if selected_stacks:

        # Ranking

        rank_list = [{"Stack": n, "Avg Health": d["health_score"].mean()} for n, d in all_data.items()]

        st.subheader("📊 Performance Ranking")

        st.dataframe(pd.DataFrame(rank_list).sort_values("Avg Health", ascending=False), use_container_width=True)
 
        # Health Plot with Color Zones

        fig_h = go.Figure()

        for z, color, yr in [("RED", "red", [0,60]), ("AMBER", "orange", [60,80]), ("GREEN", "green", [80,100])]:

            fig_h.add_hrect(y0=yr[0], y1=yr[1], fillcolor=color, opacity=0.05, line_width=0, annotation_text=z)

        for n in selected_stacks:

            fig_h.add_trace(go.Scatter(x=all_data[n]['Time'], y=all_data[n]['health_score'], name=n))

        fig_h.update_layout(height=400, yaxis_range=[0,105], hovermode="x unified")

        st.plotly_chart(fig_h, use_container_width=True)
 
        # SUSTAINED ALERT LOG (Sensitivity: 10 points)

        st.divider()

        st.subheader("🚨 Sustained Red Alert Log")

        target = st.selectbox("Deep-Dive for Stack:", selected_stacks)

        df_target = all_data[target]

        df_target["is_red"] = (df_target["health_score"] < 60).astype(int)

        df_target["group"] = (df_target["is_red"] != df_target["is_red"].shift()).cumsum()

        red_episodes = df_target[df_target["is_red"] == 1].groupby("group").agg(

            Start=("Time", "min"), End=("Time", "max"), Points=("Time", "count")

        )

        sustained = red_episodes[red_episodes["Points"] >= 10]

        if not sustained.empty:

            st.warning(f"Detected {len(sustained)} sustained red alerts (10+ points).")

            st.table(sustained.reset_index(drop=True))

        else:

            st.success("No sustained critical alerts detected.")
 
        # Status Panel Diagnostics

        st.subheader("Root Cause Inspection")

        time_strs = df_target['Time'].dt.strftime('%H:%M:%S').tolist()

        sel_time = st.select_slider("Pick Timestamp", options=time_strs)

        row = df_target[df_target['Time'].dt.strftime('%H:%M:%S') == sel_time].iloc[0]

        zone = "GREEN" if row['health_score'] >= 80 else "AMBER" if row['health_score'] >= 60 else "RED"

        bg = "#e8f5e9" if zone == "GREEN" else "#fff3e0" if zone == "AMBER" else "#ffebee"

        brd = "#2e7d32" if zone == "GREEN" else "#ef6c00" if zone == "AMBER" else "#c62828"

        p_map = {"Voltage": row["p_v"]*40, "Cell Spread": row["p_s"]*30, "Flow": row["p_f"]*20, "Cond": row["p_c"]*10}

        active = sorted({k: v for k, v in p_map.items() if v > 0}.items(), key=lambda x: x[1], reverse=True)

        summary = f"Health drop primarily driven by <b>{active[0][0]}</b>." if active else "System performing at baseline."
 
        st.markdown(f'<div class="status-panel" style="background-color: {bg}; border-left-color: {brd};"><b>{sel_time} | Health: {row["health_score"]:.1f}</b><br>Diagnosis: {summary}</div>', unsafe_allow_html=True)

        st.table(pd.DataFrame.from_dict(p_map, orient='index', columns=['Points Subtracted']).sort_values("Points Subtracted", ascending=False))
 
# --- TAB 2: TREND ANALYSIS ---

with tab2:

    st.subheader("Comparative Relationship Plot")

    all_cols = sorted(list(set().union(*(d.columns for d in all_data.values()))))

    c1, c2 = st.columns(2)

    p1 = c1.selectbox("Left Axis Parameter", all_cols, index=all_cols.index("avg_cell_voltage") if "avg_cell_voltage" in all_cols else 0)

    p2 = c2.selectbox("Right Axis Parameter", all_cols, index=all_cols.index("stack_current") if "stack_current" in all_cols else 0)

    comp_list = st.multiselect("Stacks to plot:", list(all_data.keys()), default=list(all_data.keys())[:1])

    if comp_list:

        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

        for n in comp_list:

            if p1 in all_data[n].columns:

                fig_dual.add_trace(go.Scatter(x=all_data[n]['Time'], y=all_data[n][p1], name=f"{n}: {p1}"), secondary_y=False)

            if p2 in all_data[n].columns:

                fig_dual.add_trace(go.Scatter(x=all_data[n]['Time'], y=all_data[n][p2], name=f"{n}: {p2}", line=dict(dash='dot')), secondary_y=True)

        fig_dual.update_layout(height=600, hovermode="x unified")

        fig_dual.update_yaxes(title_text=p1, secondary_y=False)

        fig_dual.update_yaxes(title_text=p2, secondary_y=True)

        st.plotly_chart(fig_dual, use_container_width=True)
 