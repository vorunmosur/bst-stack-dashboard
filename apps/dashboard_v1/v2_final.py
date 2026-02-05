import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration & Branded Styling
# Official direct link to Ohmium logo
LOGO_URL = r"C:\Users\VorunMosur\Downloads\ohmiumlogo.png"

st.set_page_config(
    page_title="Ohmium Stack Performance Dashboard", 
    page_icon=LOGO_URL, 
    layout="wide"
)

# Custom CSS for Branded UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .status-panel { padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 8px solid; font-size: 1.1em; line-height: 1.5; }
    .grid-card-text { text-align: center; font-weight: bold; margin-top: -10px; margin-bottom: 20px; }
    .title-container { display: flex; align-items: center; gap: 20px; padding-bottom: 20px; }
    .title-container img { height: 60px; }
    </style>
    """, unsafe_allow_html=True)

# Branding: Sidebar Logo
st.logo(LOGO_URL, icon_image=LOGO_URL)

# 2. Universal Data Processing Engine (BST & MTS Support)
@st.cache_data
def load_and_process(file, mode):
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()
    
    # Time Parsing
    time_col = 'TimeStamp' if mode == "MTS" else next((c for c in df.columns if 'time' in c.lower()), 'Time')
    df['Time'] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=['Time']).sort_values('Time')
    
    # Column Mapping
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
        "v": df["avg_cell_voltage"].median() if "avg_cell_voltage" in df.columns else 1.5,
        "s": df["cell_spread"].median() if "cell_spread" in df.columns else 0.05,
        "f": pd.to_numeric(df.get(flow_col, 1), errors='coerce').median(),
        "c": pd.to_numeric(df.get(cond_col, 1), errors='coerce').median()
    }
    
    df["p_v"] = ((df["avg_cell_voltage"] - base["v"]) / base["v"]).clip(0).fillna(0)
    df["p_s"] = ((df["cell_spread"] - base["s"]) / base["s"]).clip(0).fillna(0)
    df["p_f"] = ((base["f"] - pd.to_numeric(df.get(flow_col, 0), errors='coerce')) / base["f"]).clip(0).fillna(0)
    df["p_c"] = ((pd.to_numeric(df.get(cond_col, 0), errors='coerce') - base["c"]) / base["c"]).clip(0).fillna(0)
    df["health_score"] = (100 - (df["p_v"]*40 + df["p_s"]*30 + df["p_f"]*20 + df["p_c"]*10)).clip(0, 100)
    return df, cell_cols

# 3. Sidebar
with st.sidebar:
    st.header("1. Data Ingestion")
    sys_mode = st.radio("System Type", ["BST", "MTS"])
    uploaded_files = st.file_uploader(f"Upload {sys_mode} CSV Files", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        st.divider()
        st.header("📊 Fleet Summary")
        st.metric("Total Stacks", len(uploaded_files))
        st.info(f"Analyzing {sys_mode} dataset")

if not uploaded_files:
    st.markdown(f'<div class="title-container"><h1>Ohmium Stack Performance Dashboard</h1></div>', unsafe_allow_html=True)
    st.info("👈 Please upload stack files to begin analysis.")
    st.stop()

# 4. Global Load
all_data = {}; all_cell_cols = {}
for f in uploaded_files:
    df, cells = load_and_process(f, sys_mode)
    all_data[f.name] = df
    all_cell_cols[f.name] = cells

st.markdown(f'<div class="title-container"><h1>Ohmium Stack Performance Dashboard</h1></div>', unsafe_allow_html=True)

# 5. UI Tabs
tab1, tab2, tab3 = st.tabs(["🏛️ Fleet View & Cell Analysis", "🔍 Deep-Dive Diagnostics", "📈 Comparative Trends"])

# --- TAB 1: FLEET VIEW ---
with tab1:
    st.subheader("Fleet Health (Distance to Ideal)")
    grid_cols = st.columns(3)
    for i, (name, df) in enumerate(all_data.items()):
        avg_h = df["health_score"].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=avg_h, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': name},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#003366"},
                   'steps': [{'range': [0, 60], 'color': "#ffebee"}, {'range': [60, 80], 'color': "#fff3e0"}, {'range': [80, 100], 'color': "#e8f5e9"}]}))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
        with grid_cols[i % 3]:
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f'<div class="grid-card-text">Gap to Ideal: {100-avg_h:.1f} pts</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Individual Cell Voltage Deviation (Reference: 1.7V)")
    target_c = st.selectbox("Select Stack for Cell Analysis", list(all_data.keys()), key="cell_target")
    df_c = all_data[target_c]; c_cols_c = all_cell_cols[target_c]
    time_strs_c = df_c['Time'].dt.strftime('%H:%M:%S').tolist()
    sel_time_c = st.select_slider("Select Inspection Timestamp", options=time_strs_c, key="cell_slider")
    row_c = df_c[df_c['Time'].dt.strftime('%H:%M:%S') == sel_time_c].iloc[0]

    # Manager's 1.7V Reference
    IDEAL_REF = 1.7
    deviations = row_c[c_cols_c] - IDEAL_REF
    fig_dev = go.Figure()
    fig_dev.add_trace(go.Bar(x=c_cols_c, y=deviations, marker_color=['#c62828' if x > 0 else '#1565c0' for x in deviations]))
    fig_dev.update_layout(title=f"Cell Voltage Deviation from 1.7V Benchmark at {sel_time_c}", yaxis_title="Deviation (V)", height=350)
    st.plotly_chart(fig_dev, use_container_width=True)

# --- TAB 2: DEEP-DIVE DIAGNOSTICS ---
with tab2:
    st.subheader("📊 Fleet Performance Ranking")
    rank_df = pd.DataFrame([{"Stack": n, "Avg Health": d["health_score"].mean()} for n, d in all_data.items()])
    st.table(rank_df.sort_values("Avg Health", ascending=False))

    st.subheader("Comparative Health Trends")
    sel_diag_stacks = st.multiselect("Compare Multiple Stacks (Health Score)", list(all_data.keys()), default=list(all_data.keys()), key="multi_health")
    
    fig_ht = go.Figure()
    for z, color, yr in [("RED", "red", [0,60]), ("AMBER", "orange", [60,80]), ("GREEN", "green", [80,100])]:
        fig_ht.add_hrect(y0=yr[0], y1=yr[1], fillcolor=color, opacity=0.05, line_width=0, annotation_text=z)
    for n in sel_diag_stacks:
        fig_ht.add_trace(go.Scatter(x=all_data[n]['Time'], y=all_data[n]['health_score'], name=n))
    fig_ht.update_layout(height=400, yaxis_range=[0,105], hovermode="x unified", title="Health Score Comparison Over Time")
    st.plotly_chart(fig_ht, use_container_width=True)

    st.divider()
    target_d = st.selectbox("Select Stack for Alert Log & Root Cause Inspection", list(all_data.keys()), key="diag_target")
    df_d = all_data[target_d]

    # Sustained Red Alert Log
    st.subheader(f"🚨 Sustained Red Alert Log: {target_d}")
    df_d["is_red"] = (df_d["health_score"] < 60).astype(int)
    df_d["grp"] = (df_d["is_red"] != df_d["is_red"].shift()).cumsum()
    red_eps = df_d[df_d["is_red"] == 1].groupby("grp").agg(Start=("Time", "min"), End=("Time", "max"), Count=("Time", "count"))
    sustained = red_eps[red_eps["Count"] >= 10]
    if not sustained.empty: st.warning("Sustained critical alerts detected:"); st.table(sustained.reset_index(drop=True))
    else: st.success("No sustained critical alerts.")

    st.subheader(f"Root Cause Inspection: {target_d}")
    time_strs_d = df_d['Time'].dt.strftime('%H:%M:%S').tolist()
    sel_time_d = st.select_slider("Pick Timestamp (Root Cause)", options=time_strs_d, key="rc_slider")
    row_d = df_d[df_d['Time'].dt.strftime('%H:%M:%S') == sel_time_d].iloc[0]
    
    zone = "GREEN" if row_d['health_score'] >= 80 else "AMBER" if row_d['health_score'] >= 60 else "RED"
    bg = "#e8f5e9" if zone == "GREEN" else "#fff3e0" if zone == "AMBER" else "#ffebee"
    brd = "#2e7d32" if zone == "GREEN" else "#ef6c00" if zone == "AMBER" else "#c62828"
    p_map = {"Voltage": row_d["p_v"]*40, "Spread": row_d["p_s"]*30, "Flow": row_d["p_f"]*20, "Cond": row_d["p_c"]*10}
    active = sorted({k: v for k, v in p_map.items() if v > 0}.items(), key=lambda x: x[1], reverse=True)
    summary = f"Health drop primarily driven by <b>{active[0][0]}</b>." if active else "Stack operating at baseline."
    st.markdown(f'<div class="status-panel" style="background-color: {bg}; border-left-color: {brd};"><b>{sel_time_d} | Health: {row_d["health_score"]:.1f}</b><br>Diagnosis: {summary}</div>', unsafe_allow_html=True)
    
    csv = pd.DataFrame.from_dict(p_map, orient='index', columns=['Points Subtracted']).to_csv().encode('utf-8')
    st.download_button("Download Report (CSV)", data=csv, file_name=f"diag_{target_d}.csv", mime='text/csv')

# --- TAB 3: COMPARATIVE TRENDS ---
with tab3:
    st.subheader("Dual-Parameter Comparative Relationship Plot")
    all_p = ["None"] + sorted(list(set().union(*(d.columns for d in all_data.values()))))
    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Left Axis Parameter", [p for p in all_p if p != "None"], index=all_p.index("avg_cell_voltage") if "avg_cell_voltage" in all_p else 0)
    p2 = c2.selectbox("Right Axis Parameter", all_p, index=0)
    comp_list = st.multiselect("Stacks to Plot Relationships:", list(all_data.keys()), default=list(all_data.keys())[:1], key="trend_stacks")
    
    if comp_list:
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        for n in comp_list:
            if p1 in all_data[n].columns:
                fig_dual.add_trace(go.Scatter(x=all_data[n]['Time'], y=all_data[n][p1], name=f"{n}: {p1}"), secondary_y=False)
            if p2 != "None" and p2 in all_data[n].columns:
                fig_dual.add_trace(go.Scatter(x=all_data[n]['Time'], y=all_data[n][p2], name=f"{n}: {p2}", line=dict(dash='dot')), secondary_y=True)
        fig_dual.update_layout(height=600, hovermode="x unified")
        fig_dual.update_yaxes(title_text=f"<b>{p1}</b>", secondary_y=False)
        if p2 != "None": fig_dual.update_yaxes(title_text=f"<b>{p2}</b> (Dashed)", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)
        