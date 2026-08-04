import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Tigray IDF & Hydrologic Design Engine",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .main-header { font-size: 24px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-header { font-size: 14px; color: #4B5563; margin-bottom: 20px; }
    .coord-box {
        background-color: #EBF8FF;
        border-left: 4px solid #3182CE;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .status-box {
        background-color: #1A202C;
        color: #FFFFFF;
        padding: 8px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to convert decimal degrees to DMS
def dec_to_dms(deg, is_lat=True):
    d = int(abs(deg))
    m = int((abs(deg) - d) * 60)
    s = (abs(deg) - d - m / 60) * 3600
    direction = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    return f"{d}°{m:02d}'{s:04.1f}\"{direction}"

# ==========================================
# 2. SIDEBAR - LOCATION SELECTOR
# ==========================================
st.sidebar.markdown("### 📍 Location Selector")

zone_option = st.sidebar.selectbox(
    "Select Zone",
    ["Northwestern Zone", "Central Zone", "Eastern Zone", "Southern Zone"],
    index=0
)

woreda_option = st.sidebar.selectbox(
    "Select Woreda Hub / Town",
    ["Shire", "Axum", "Adwa", "Sheraro"],
    index=0
)

# Shire Focus Point Coordinates
lat_dec, lon_dec = 14.1020, 38.2830
utm_easting, utm_northing = 421784.5, 1558431.6
lat_dms = dec_to_dms(lat_dec, is_lat=True)
lon_dms = dec_to_dms(lon_dec, is_lat=False)

st.sidebar.markdown(f"""
<div class="coord-box">
    <b>Focus Point Coordinates (Shire):</b><br>
    <b>DMS:</b> Lat: {lat_dms} | Lon: {lon_dms}<br>
    <b>UTM Zone 37N:</b> E {utm_easting:,.1f} m, N {utm_northing:,.1f} m<br>
    <small><b>Decimal:</b> Lat: {lat_dec:.4f}°N | Lon: {lon_dec:.4f}°E</small>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="status-box">🔒 DATA STATUS: READ-ONLY ACCESS</div>', unsafe_allow_html=True)

# ==========================================
# 3. SYNTHETIC IDF & FREQUENCY DATABASE
# ==========================================
durations = np.array([5, 10, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 240, 360, 720, 1440])
return_periods = [2, 5, 10, 25, 50, 100]

# Empirical IDF parameters for Shire region (Sherman Parameter Model: I = a / (t + b)^c)
idf_data = {}
for T in return_periods:
    a = 450 * (T ** 0.22)
    b = 12.0
    c = 0.78
    intensities = a / ((durations + b) ** c)
    idf_data[f"T = {T} Years"] = intensities

idf_df = pd.DataFrame(idf_data, index=durations)
idf_df.index.name = "Duration_min"

# ==========================================
# 4. MAIN LAYOUT & TITLE
# ==========================================
st.title("🌧️ Tigray Regional IDF & Hydrologic Storm Engine")
st.markdown("Dual-visualization & Custom Design Storm Computation sampled at 5-minute to 24-hour intervals for micro-catchment hydraulic design.")

# ==========================================
# 5. DISPLAYS 1, 2, AND 3: IDF VISUALIZATIONS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Linear IDF Plot (0-120 min)",
    "2️⃣ Semi-Logarithmic IDF Plot",
    "3️⃣ Log-Log IDF Plot",
    "4️⃣ Custom Design Storm Compute"
])

colors = {
    "T = 2 Years": "#1f77b4",
    "T = 5 Years": "#ff7f0e",
    "T = 10 Years": "#2ca02c",
    "T = 25 Years": "#d62728",
    "T = 50 Years": "#9467bd",
    "T = 100 Years": "#8c564b"
}

# --- DISPLAY 1: LINEAR IDF PLOT (0-120 min) ---
with tab1:
    st.subheader("1. IDF Curve Plot (Linear plot for high-intensity zone 0-120 min)-Shire")
    
    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    short_mask = durations <= 120
    d_short = durations[short_mask]
    
    for col in idf_df.columns:
        ax1.plot(
            d_short, 
            idf_df.loc[d_short, col], 
            marker='o', 
            linestyle='--', 
            linewidth=1.5, 
            markersize=4, 
            color=colors[col], 
            label=col
        )
    
    ax1.set_xlim(0, 120)
    ax1.set_xticks(np.arange(0, 121, 15))
    ax1.set_xlabel("Storm Duration (minutes)", fontsize=10)
    ax1.set_ylabel("Rainfall Intensity (mm/hr)", fontsize=10)
    ax1.set_title("1. IDF Curve Plot (Linear plot for high-intensity zone 0-120 min)-Shire", fontsize=11, fontweight='bold')
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(title="Return Period", loc="upper right")
    
    st.pyplot(fig1)

# --- DISPLAY 2: SEMI-LOGARITHMIC IDF PLOT ---
with tab2:
    st.subheader("2. IDF Curve Plot (Semilogarithmic plot for extended duration spectrum 5 min - 24 hrs)-Shire")
    
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
    
    for col in idf_df.columns:
        ax2.semilogx(
            durations, 
            idf_df[col], 
            marker='s', 
            linestyle='-', 
            linewidth=1.5, 
            markersize=4, 
            color=colors[col], 
            label=col
        )
    
    ax2.set_xlim(5, 1440)
    ax2.set_xlabel("Storm Duration (minutes)", fontsize=10)
    ax2.set_ylabel("Rainfall Intensity (mm/hr)", fontsize=10)
    ax2.set_title("2. IDF Curve Plot (Semilogarithmic plot for extended duration spectrum 5 min - 24 hrs)-Shire", fontsize=11, fontweight='bold')
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(title="Return Period", loc="upper right")
    
    st.pyplot(fig2)

# --- DISPLAY 3: BOTH X,Y LOGARITHMIC PLOT ---
with tab3:
    st.subheader("3. IDF Curve Plot (Double-Logarithmic plot for full spectrum 5 min - 24 hrs)-Shire")
    
    fig3, ax3 = plt.subplots(figsize=(10, 5), dpi=150)
    
    for col in idf_df.columns:
        ax3.loglog(
            durations, 
            idf_df[col], 
            marker='^', 
            linestyle='-', 
            linewidth=1.5, 
            markersize=4, 
            color=colors[col], 
            label=col
        )
    
    ax3.set_xlim(5, 1440)
    ax3.set_xlabel("Storm Duration (minutes)", fontsize=10)
    ax3.set_ylabel("Rainfall Intensity (mm/hr)", fontsize=10)
    ax3.set_title("3. IDF Curve Plot (Double-Logarithmic plot for full spectrum 5 min - 24 hrs)-Shire", fontsize=11, fontweight='bold')
    ax3.grid(True, which="both", linestyle=":", alpha=0.6)
    ax3.legend(title="Return Period", loc="upper right")
    
    st.pyplot(fig3)

# ==========================================
# 6. DISPLAY 4: USER PREFERENCE COMPUTING
# ==========================================
with tab4:
    st.subheader("4. Standardized User-Preference Design Storm Generator")
    st.markdown("Select critical hydrologic parameters to compute design storm hyetographs and net effective depth.")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        user_loc = st.selectbox("Selected Target Location", ["Shire", "Axum", "Adwa", "Sheraro"], index=0)
        user_return = st.selectbox("Return Period (T, Years)", [2, 5, 10, 25, 50, 100], index=3)
        
    with col_b:
        user_duration = st.number_input("Storm Duration (minutes)", min_value=15, max_value=1440, value=120, step=15)
        time_step = st.selectbox("Temporal Resolution (dt, min)", [5, 10, 15], index=0)
        
    with col_c:
        dist_model = st.selectbox("Frequency Distribution Model", ["Gumbel (EV-I)", "Log-Pearson Type III", "GEV"], index=0)
        scs_type = st.selectbox("SCS Hyetograph Synthetic Type", ["Type II (Standard)", "Type I", "Type IA", "Type III"], index=0)
        
    with col_d:
        cn_value = st.number_input("Curve Number (CN)", min_value=30, max_value=100, value=85)
        compute_btn = st.button("⚡ Compute Design Storm", type="primary", use_container_width=True)

    if compute_btn or "storm_computed" in st.session_state:
        st.session_state["storm_computed"] = True
        
        a_param = 450 * (user_return ** 0.22)
        calc_intensity = a_param / ((user_duration + 12.0) ** 0.78)
        calc_depth = calc_intensity * (user_duration / 60.0)
        
        S = (25400 / cn_value) - 254
        Ia = 0.2 * S
        if calc_depth > Ia:
            P_eff = ((calc_depth - Ia) ** 2) / (calc_depth - Ia + S)
        else:
            P_eff = 0.0
            
        st.markdown("---")
        st.markdown(f"### 📊 Design Storm Summary Results for **{user_loc}** ($T = {user_return}$ Years)")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Design Intensity (mm/hr)", f"{calc_intensity:.2f} mm/hr")
        m2.metric("Total Rainfall Depth (P)", f"{calc_depth:.2f} mm")
        m3.metric("Initial Abstraction (Ia)", f"{Ia:.2f} mm")
        m4.metric("Effective Depth (P_eff)", f"{P_eff:.2f} mm")
        
        n_blocks = int(user_duration / time_step)
        t_arr = np.arange(time_step, user_duration + time_step, time_step)
        
        p_cum = [a_param / ((t + 12.0) ** 0.78) * (t / 60.0) for t in t_arr]
        p_inc = np.diff(np.insert(p_cum, 0, 0))
        
        p_block = np.zeros(n_blocks)
        sorted_inc = np.sort(p_inc)[::-1]
        center = n_blocks // 2
        
        p_block[center] = sorted_inc[0]
        left, right = center - 1, center + 1
        for i in range(1, n_blocks):
            if i % 2 == 1 and left >= 0:
                p_block[left] = sorted_inc[i]
                left -= 1
            elif right < n_blocks:
                p_block[right] = sorted_inc[i]
                right += 1
                
        storm_df = pd.DataFrame({
            "Time (min)": t_arr,
            "Incremental Depth (mm)": np.round(p_block, 2),
            "Intensity (mm/hr)": np.round(p_block / (time_step / 60.0), 2),
            "Cumulative Depth (mm)": np.round(np.cumsum(p_block), 2)
        })
        
        fig_storm, ax_storm = plt.subplots(figsize=(9, 4), dpi=150)
        ax_storm.bar(t_arr, storm_df["Intensity (mm/hr)"], width=time_step*0.8, color="#3182CE", alpha=0.8, label="Hyetograph Intensity (mm/hr)")
        ax_storm2 = ax_storm.twinx()
        ax_storm2.plot(t_arr, storm_df["Cumulative Depth (mm)"], color="#E53E3E", linewidth=2, label="Cumulative Mass Curve (mm)")
        
        ax_storm.set_xlabel("Time (minutes)")
        ax_storm.set_ylabel("Intensity (mm/hr)", color="#3182CE")
        ax_storm2.set_ylabel("Cumulative Depth (mm)", color="#E53E3E")
        ax_storm.set_title(f"Design Hyetograph & Mass Curve - {user_duration} min ({user_return}-Yr Return Period)")
        ax_storm.grid(True, linestyle=":", alpha=0.5)
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.pyplot(fig_storm)
        with c2:
            st.dataframe(storm_df, height=300, use_container_width=True)

# ==========================================
# 7. REGIONAL DATA ACCESS NOTICE
# ==========================================
st.markdown("---")
st.subheader("📁 Regional Hydrological Database Access")
st.info(
    "Complete regional hydrological matrices, baseline frequency datasets, and exportable high-resolution spreadsheets "
    "are available for academic research and professional engineering design upon formal email request."
)
