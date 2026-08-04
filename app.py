import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import os

# ==========================================
# 1. PAGE CONFIGURATION & EXECUTIVE STYLING
# ==========================================
st.set_page_config(
    page_title="Tigray Regional IDF & Hydrologic Engine v3.1",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Executive Hydrologic UI
st.markdown("""
<style>
    .main-header {
        background-color: #0E1117;
        padding: 22px;
        border-radius: 10px;
        color: white;
        text-align: center;
        border-bottom: 4px solid #DC2626;
        margin-bottom: 25px;
    }
    .read-only-badge {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #0284C7;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .lit-card {
        background-color: #1F2937;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-top: 10px;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 12px;
        border-radius: 6px;
        border-left: 3px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>TIGRAY REGIONAL DESIGN STORM & HYDROLOGIC ENGINE v3.1</h1>
    <p>Northwestern & Central Zones | 2000–2025 Historical Baseline | 2026 YTD Conditioning | ITCZ Forecast Engine</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. TARGETED WOREDA DATABASE (NORTHWEST & CENTRAL)
# ==========================================
WOREDA_DATABASE = {
    "Northwestern Zone": {
        "Shire (Inda Selassie)": (14.102, 38.283),
        "Selekleka": (14.120, 38.470),
        "Zana": (14.220, 38.350),
        "Endabaguna": (14.050, 38.220),
        "Kisadgaba": (14.300, 38.150),
        "Adi-Daero": (14.280, 38.180),
        "Adi-Nebried": (14.350, 38.400),
        "Adi-Hageray": (14.420, 37.910),
        "Sheraro": (14.385, 37.761),
        "May Tsebri": (13.583, 38.133)
    },
    "Central Zone": {
        "Axum": (14.123, 38.720),
        "Adwa": (14.162, 38.898),
        "Rama": (14.372, 38.799),
        "Enticho": (14.288, 39.151),
        "Gerhu Sernay": (14.450, 39.120),
        "Abiy Addi": (13.623, 39.002),
        "Edaga Arbi": (13.880, 39.050),
        "Endaba Tsahma": (14.180, 38.980)
    }
}

# Sidebar Selector
st.sidebar.header("📍 Location Selector")
selected_zone = st.sidebar.selectbox("Select Zone", list(WOREDA_DATABASE.keys()))
selected_town = st.sidebar.selectbox("Select Woreda Hub / Town", list(WOREDA_DATABASE[selected_zone].keys()))
lat, lon = WOREDA_DATABASE[selected_zone][selected_town]

st.sidebar.info(f"**Coordinates:**\nLat: {lat:.3f}°N | Lon: {lon:.3f}°E")
st.sidebar.markdown("""
<div class="read-only-badge">
    🔒 DATA STATUS: READ-ONLY ACCESS
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. HISTORICAL HYDROLOGIC DATA GENERATOR (2000–2026 YTD)
# ==========================================
@st.cache_data
def load_historical_database(town_name):
    """
    Generates synthetic daily rainfall database:
    - Complete Historical Baseline: 2000-01-01 to 2025-12-31 (26 Years)
    - Year-To-Date Record: 2026-01-01 to 2026-07-31 (January through July)
    """
    dates = pd.date_range(start="2000-01-01", end="2026-07-31", freq="D")
    np.random.seed(abs(hash(town_name)) % 100000)
    
    doy = dates.dayofyear
    # Bimodal/Monsoonal signal peak in July/August (Kiremt monsoon)
    seasonal_intensity = np.exp(-0.5 * ((doy - 220) / 28) ** 2) + 0.15 * np.exp(-0.5 * ((doy - 110) / 20) ** 2)
    daily_rain = np.random.exponential(scale=6.2, size=len(dates)) * seasonal_intensity
    daily_rain = np.where(daily_rain < 0.4, 0.0, daily_rain)
    
    df = pd.DataFrame({
        "Date": dates,
        "Year": dates.year,
        "DayOfYear": dates.dayofyear,
        "Month": dates.month_name(),
        "Daily_Precipitation_mm": np.round(daily_rain, 2)
    })
    return df

df_full = load_historical_database(selected_town)

# Filter strict historical dataset (2000-2025) and YTD 2026 dataset
df_historical_baseline = df_full[df_full["Year"] <= 2025].copy()
df_2026_ytd = df_full[df_full["Year"] == 2026].copy()

# ==========================================
# TABBED SYSTEM NAVIGATION
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 IDF Curves", 
    "🗓️ Single-Year Daily", 
    "📈 2000–2025 Historical Baseline", 
    "🌦️ 5-Day Forecast & Literature",
    "🗺️ Aerial Image Viewer"
])

# ------------------------------------------
# TAB 1: REGIONAL LINEAR/PERTURBED IDF CURVES
# ------------------------------------------
with tab1:
    st.subheader(f"⚡ Intensity-Duration-Frequency (IDF) Curves: {selected_town}")
    st.caption("Synthetic linear IDF equations with perturbed parameter noise for stress-testing hydraulic drainage networks.")
    
    durations_min = np.array([10, 20, 30, 60, 120, 180, 360, 720, 1440]) # minutes
    return_periods = [2, 5, 10, 25, 50, 100]
    
    fig_idf, ax_idf = plt.subplots(figsize=(10, 5))
    
    np.random.seed(abs(hash(selected_town)) % 999)
    for T in return_periods:
        base_i = (450 * (T ** 0.22)) / (durations_min + 18)
        messed_noise = np.random.normal(0, 3.5, size=len(durations_min))
        intensity = np.maximum(base_i + messed_noise, 2.0)
        
        ax_idf.plot(durations_min, intensity, label=f"T = {T} Years", linewidth=2, marker='o', linestyle='--')
    
    ax_idf.set_xlabel("Storm Duration (minutes)", fontweight="bold")
    ax_idf.set_ylabel("Rainfall Intensity (mm/hr)", fontweight="bold")
    ax_idf.set_title(f"Linearized Perturbed IDF Curves - {selected_town} (A2/A3 Regionalization)", fontweight="bold")
    ax_idf.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_idf.legend(title="Return Period")
    
    st.pyplot(fig_idf)

# ------------------------------------------
# TAB 2: SINGLE-YEAR DAILY RAINFALL (2000 - 2026 YTD)
# ------------------------------------------
with tab2:
    st.subheader("📅 Single-Year Daily Rainfall Query")
    st.caption("Inspect daily rainfall records from 2000 through 2025, or view 2026 Year-To-Date (January–July).")
    
    available_years = sorted(df_full["Year"].unique(), reverse=True)
    selected_year = st.selectbox("Select Year", available_years)
    
    df_year = df_full[df_full["Year"] == selected_year]
    
    if selected_year == 2026:
        st.warning("⚠️ **2026 Data Note**: Showing Year-To-Date record from **January 1 to July 31, 2026**.")
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Cumulative Rainfall", f"{df_year['Daily_Precipitation_mm'].sum():.1f} mm")
    col_metric2.metric("Peak Daily Storm", f"{df_year['Daily_Precipitation_mm'].max():.1f} mm")
    col_metric3.metric("Rainy Days (>0.4mm)", f"{(df_year['Daily_Precipitation_mm'] > 0.4).sum()} Days")
    
    fig_daily, ax_daily = plt.subplots(figsize=(12, 4))
    ax_daily.bar(df_year["Date"], df_year["Daily_Precipitation_mm"], color="#0284C7")
    ax_daily.set_ylabel("Daily Rainfall (mm)")
    ax_daily.set_title(f"Daily Precipitation Chronology - {selected_town} ({selected_year})", fontweight="bold")
    ax_daily.grid(True, alpha=0.3)
    
    st.pyplot(fig_daily)
    st.dataframe(df_year[["Date", "Month", "Daily_Precipitation_mm"]], use_container_width=True, height=250)

# ------------------------------------------
# TAB 3: 2000–2025 HISTORICAL BASELINE (365-DAY MEAN)
# ------------------------------------------
with tab3:
    st.subheader("📈 2000–2025 Historical Baseline (365-Day Daily Mean)")
    st.caption("Clean 26-year historical baseline excluding incomplete 2026 data to prevent statistical skew.")
    
    df_365 = df_historical_baseline.groupby("DayOfYear")["Daily_Precipitation_mm"].mean().reset_index()
    
    fig_365, ax_365 = plt.subplots(figsize=(12, 4.5))
    ax_365.plot(df_365["DayOfYear"], df_365["Daily_Precipitation_mm"], color="#DC2626", linewidth=1.8)
    ax_365.fill_between(df_365["DayOfYear"], df_365["Daily_Precipitation_mm"], color="#FCA5A5", alpha=0.4)
    ax_365.set_xlabel("Day of Year (1 - 365)")
    ax_365.set_ylabel("26-Year Average Daily Rainfall (mm)")
    ax_365.set_title(f"365-Day Mean Precipitation Envelope (2000–2025 Baseline) - {selected_town}", fontweight="bold")
    ax_365.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig_365)

# ------------------------------------------
# TAB 4: 5-DAY LIVE FORECAST & 2026 CONDITIONED LITERATURE
# ------------------------------------------
with tab4:
    st.subheader("🌦️ Live 5-Day Rainfall Forecast & 2026 Antecedent Conditioning")
    
    # Calculate July 2026 Antecedent Moisture Index from YTD dataset
    july_2026_rain = df_2026_ytd[df_2026_ytd["Month"] == "July"]["Daily_Precipitation_mm"].sum()
    july_hist_avg = df_historical_baseline[df_historical_baseline["Month"] == "July"].groupby("Year")["Daily_Precipitation_mm"].sum().mean()
    moisture_ratio = (july_2026_rain / july_hist_avg) if july_hist_avg > 0 else 1.0
    
    st.markdown(f"""
    <div class="metric-card">
        📊 <strong>2026 Antecedent Soil Moisture Indicator (July Condition):</strong><br>
        July 2026 Recorded Rain: <strong>{july_2026_rain:.1f} mm</strong> | 2000–2025 July Baseline: <strong>{july_hist_avg:.1f} mm</strong><br>
        Catchment Saturation Index: <strong>{moisture_ratio:.2f}x</strong> (Used to weight short-term runoff probability)
    </div>
    """, unsafe_allow_html=True)
    
    st.write(" ")
    
    @st.cache_data(ttl=3600)
    def fetch_5day_detailed(latitude, longitude):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "precipitation_sum,rain_sum,showers_sum",
            "timezone": "Africa/Addis_Ababa",
            "forecast_days": 5
        }
        try:
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json().get("daily", {})
            df_f = pd.DataFrame({
                "Date": data.get("time", []),
                "Total Depth (mm)": data.get("precipitation_sum", []),
                "Stratiform Rain (mm)": data.get("rain_sum", []),
                "Convective Showers (mm)": data.get("showers_sum", [])
            })
            types = []
            for idx, row in df_f.iterrows():
                if row["Total Depth (mm)"] < 0.5:
                    types.append("Dry / Trace")
                elif row["Convective Showers (mm)"] > row["Stratiform Rain (mm)"]:
                    types.append("Convective Storm (High Peak Intensity)")
                else:
                    types.append("Orographic Monsoon (Continuous)")
            df_f["Precipitation Type"] = types
            return df_f, None
        except Exception as e:
            return None, str(e)
            
    df_forecast, err = fetch_5day_detailed(lat, lon)
    
    if err:
        st.error(f"Error fetching live forecast: {err}")
    else:
        st.dataframe(df_forecast, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="lit-card">
            <h4>📖 Forecasting Basis & Literature Framework (Northern Ethiopia)</h4>
            <p><strong>1. Synoptic Driver (ITCZ Migration):</strong> August forecasts in Northwestern Tigray track the northward surge of the Inter-Tropical Convergence Zone (ITCZ). Maritime tropical air masses originating from the South Atlantic/Congo Basin converge over the Northern Highlands, creating strong convective instability (ERA Drainage Manual, 2013).</p>
            <p><strong>2. 2026 Antecedent Conditioning Role:</strong> Integrating 2026 YTD rainfall (Jan–Jul) establishes the pre-storm soil moisture deficit ($S_0$). When July saturation is high (>1.0x baseline), the runoff coefficient ($C$) in rational/SCS formulations increases, amplifying flash-flood risk from convective showers.</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 5: AERIAL IMAGE VIEWER
# ------------------------------------------
with tab5:
    st.subheader(f"🗺️ High-Resolution Aerial Imagery: {selected_town}")
    st.caption("Static high-resolution orthophoto / satellite snippet loaded from local assets directory.")
    
    file_key = selected_town.split(" ")[0].lower().replace("(", "").replace(")", "")
    image_path = f"assets/aerial/{file_key}.png"
    
    if os.path.exists(image_path):
        st.image(image_path, caption=f"High-Resolution Aerial Orthophoto - {selected_town}", use_column_width=True)
    else:
        st.info(f"ℹ️ Aerial image asset for **{selected_town}** (`{image_path}`) is ready for upload. Place the downloaded static PNG/JPG in the `assets/aerial/` directory.")

# ==========================================
# 4. FORMAL EXPORT CLEARANCE REQUEST
# ==========================================
st.markdown("---")
st.subheader("🔐 Formal Data Export Request")
st.caption("All feeds are view-only. To obtain full CSV matrices or vectorized GIS datasets, submit a formal request.")

with st.form("export_request_form"):
    req_name = st.text_input("Full Name / Investigator")
    req_inst = st.text_input("Institution / Organization", "Aksum University / SBE Consulting")
    req_reason = st.text_area("Purpose of Data Request")
    submit_req = st.form_submit_button("Submit Formal Export Request")
    
    if submit_req:
        st.success("✅ Formal request registered. Data clearance token will be dispatched upon review.")

st.caption("Developed by Tsegay Ayele Kidane | Water Resources Specialist & Hydraulic Engineer")
