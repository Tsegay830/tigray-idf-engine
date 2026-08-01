import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Tigray Regional IDF Engine v1.0",
    page_icon="🌧️",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Header Dark Banner */
    .header-box {
        background-color: #0b192c;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 25px;
    }
    .header-title {
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .header-subtitle {
        color: #00adb5;
        font-size: 13px;
        font-weight: 600;
    }

    /* Result Metric Card */
    .result-card {
        background-color: #f8f9fa;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .result-header {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-value {
        color: #0056b3;
        font-size: 34px;
        font-weight: 800;
        margin: 8px 0;
    }
    .result-depth {
        color: #334155;
        font-size: 14px;
        font-weight: 500;
    }

    /* Footer Banner */
    .footer-box {
        background-color: #0b192c;
        border-radius: 6px;
        padding: 12px 20px;
        margin-top: 30px;
    }
    .footer-text {
        color: #cbd5e1;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- HYDROLOGIC ENGINE DATA ---
DATA = {
    'Mekelle': {'a': 1420.5, 'b': 14.2, 'c': 0.81, 'm': 0.18},
    'Adigrat': {'a': 1280.0, 'b': 12.5, 'c': 0.79, 'm': 0.19},
    'Axum':    {'a': 1350.2, 'b': 13.8, 'c': 0.80, 'm': 0.185},
    'Shire':   {'a': 1490.8, 'b': 15.1, 'c': 0.82, 'm': 0.175},
    'Alamata': {'a': 1510.4, 'b': 16.0, 'c': 0.83, 'm': 0.170}
}

DURATIONS_MIN = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]

def format_duration(minutes):
    if minutes < 60:
        return f"{minutes} min"
    elif minutes % 60 == 0:
        hrs = minutes // 60
        return f"{hrs} hr ({minutes} min)" if hrs == 1 else f"{hrs} hrs"
    else:
        return f"{minutes / 60:.1f} hrs"

def calculate_intensity(city, T, t_min):
    p = DATA[city]
    return (p['a'] * (T ** p['m'])) / ((t_min + p['b']) ** p['c'])

# --- 1. HEADER BANNER ---
st.markdown("""
<div class="header-box">
    <div class="header-title">TIGRAY REGIONAL DESIGN STORM ENGINE v1.0</div>
    <div class="header-subtitle">Developed by: Tsegay Ayele Kidane | Hydraulic Engineer | Water Resources Specialist | Water Systems Researcher</div>
</div>
""", unsafe_allow_html=True)

# --- 2. MAIN TWO-COLUMN LAYOUT ---
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown("### ⚙️ **Design Criteria Controls**")
        city = st.selectbox("Select City / Urban Hub:", list(DATA.keys()), index=3)
        T = st.selectbox("Return Period (T in Years):", RETURN_PERIODS, index=3)
        
        duration_options = {format_duration(d): d for d in DURATIONS_MIN}
        selected_label = st.selectbox("Storm Duration (t):", list(duration_options.keys()), index=3)
        t_min = duration_options[selected_label]

        intensity = calculate_intensity(city, T, t_min)
        depth = intensity * (t_min / 60.0)

        st.button("⚡ COMPUTE DESIGN STORM", use_container_width=True, type="primary")

with col_right:
    with st.container(border=True):
        st.markdown("### 📊 **Hydrologic Output Export**")
        st.markdown(f"**Selected Location:** `{city}`")
        st.markdown(f"**Design Event:** `T = {T}-Year | t = {selected_label}`")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">DESIGN RAINFALL INTENSITY</div>
            <div class="result-value">{intensity:.2f} mm/hr</div>
            <div class="result-depth">Effective Depth: <b>{depth:.2f} mm</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Summary Matrix Setup
        matrix = []
        for t in DURATIONS_MIN:
            row = {'Duration': format_duration(t)}
            for rep_T in RETURN_PERIODS:
                row[f'T={rep_T}yr (mm/hr)'] = round(calculate_intensity(city, rep_T, t), 2)
            matrix.append(row)
        df = pd.DataFrame(matrix)

        st.download_button(
            label="💾 Export IDF Summary Matrix (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"{city}_IDF_Summary_Matrix.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- 3. AUTOMATIC INTERACTIVE IDF PLOTTING WINDOW ---
st.write("")
with st.container(border=True):
    st.markdown("### 📈 **Automatic IDF Plotting Window**")
    
    # Styled like standalone Matplotlib Figure (Figure 1 window style)
    fig, ax = plt.subplots(figsize=(11, 5))
    durations = np.array(DURATIONS_MIN)

    for rep_T in RETURN_PERIODS:
        intensities = [calculate_intensity(city, rep_T, t) for t in durations]
        ax.plot(durations, intensities, marker='o', markersize=4, linewidth=1.5, label=f'T={rep_T}yr')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"{city} Catchment - Intensity-Duration-Frequency (IDF) Curves\nAuthored by: Tsegay Ayele Kidane", fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Duration (Minutes)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Rainfall Intensity (mm/hr)', fontsize=10, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(title="Return Period", frameon=True, loc="upper right")
    plt.tight_layout()
    
    st.pyplot(fig)

# --- 4. FOOTER BANNER & CONTACT ---
st.markdown("""
<div class="footer-box">
    <span class="footer-text">🔒 <b>Raw Daily Data:</b> Restricted Backend Vault | © Tsegay Ayele Kidane</span>
</div>
""", unsafe_allow_html=True)

st.write("")
if st.button("Request 26-Yr Daily Series Access"):
    st.info("please submit your request to Tsegay Ayele Kidane via Phone:+251929112551 or via email: amen.atsegay@gmail.com")
