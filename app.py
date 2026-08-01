import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Tigray Regional IDF Engine",
    page_icon="🌧️",
    layout="wide"
)

# Engine Parameters
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

# Header
st.title("🌧️ TIGRAY REGIONAL DESIGN STORM ENGINE v1.0")
st.caption("Developed & Authored by: **Tsegay Ayele Kidane** | Water Resources Specialist")

st.divider()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("⚙️ Design Criteria Controls")
    city = st.selectbox("Select City / Urban Hub", list(DATA.keys()))
    T = st.selectbox("Return Period (T in Years)", RETURN_PERIODS, index=1)
    
    duration_options = {format_duration(d): d for d in DURATIONS_MIN}
    selected_label = st.selectbox("Storm Duration (t)", list(duration_options.keys()))
    t_min = duration_options[selected_label]

    intensity = calculate_intensity(city, T, t_min)
    depth = intensity * (t_min / 60.0)

with col2:
    st.subheader("📊 Hydrologic Outputs")
    st.metric(label="Design Rainfall Intensity", value=f"{intensity:.2f} mm/hr")
    st.metric(label="Effective Storm Depth", value=f"{depth:.2f} mm")

    st.subheader("💾 Summary Matrix Export")
    matrix = []
    for t in DURATIONS_MIN:
        row = {'Duration': format_duration(t)}
        for rep_T in RETURN_PERIODS:
            row[f'T={rep_T}yr (mm/hr)'] = round(calculate_intensity(city, rep_T, t), 2)
        matrix.append(row)
    df = pd.DataFrame(matrix)
    
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label="Download IDF Matrix (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"{city}_IDF_Summary_Matrix.csv",
        mime="text/csv"
    )

st.divider()

# Interactive Plot
st.subheader("📈 Interactive IDF Curves")
fig, ax = plt.subplots(figsize=(10, 4.5))
durations = np.array(DURATIONS_MIN)

for rep_T in RETURN_PERIODS:
    intensities = [calculate_intensity(city, rep_T, t) for t in durations]
    ax.plot(durations, intensities, marker='o', label=f'T={rep_T}yr')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Duration (Minutes)', fontweight='bold')
ax.set_ylabel('Rainfall Intensity (mm/hr)', fontweight='bold')
ax.set_title(f'{city} Catchment - Intensity-Duration-Frequency (IDF) Curves', fontweight='bold')
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(title="Return Period")
st.pyplot(fig)

# Footer Vault Section
st.divider()
st.info("🔒 **Raw Daily Data Vault:** Access to the 26-year daily rainfall series requires formal research/project clearance.")
if st.button("Request 26-Yr Daily Series"):
    st.success("To request access, please contact Tsegay Ayele Kidane with your institutional project proposal.")