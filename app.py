import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Tigray Regional IDF & Hydrologic Engine",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive UI styling
st.markdown("""
<style>
    .main-header {
        background-color: #0E1117;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        border-bottom: 3px solid #FF4B4B;
        margin-bottom: 25px;
    }
    .stMetric {
        background-color: #1F2937;
        padding: 15px;
        border-radius: 8px;
    }
    .gate-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>TIGRAY REGIONAL DESIGN STORM & HYDROLOGIC ENGINE v2.0</h1>
    <p>Integrated IDF Modeling, Daily Data Processing, and Live 5-Day Precipitation Forecasting</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. EXTENDED WOREDA TOWNS & COORDINATES
# ==========================================
WOREDA_DATABASE = {
    "Central Zone": {
        "Axum": (14.123, 38.720),
        "Adwa": (14.162, 38.898),
        "Abiy Addi": (13.623, 39.002),
        "Rama": (14.372, 38.799),
        "Enticho": (14.288, 39.151)
    },
    "Eastern Zone": {
        "Adigrat": (14.277, 39.462),
        "Wukro": (13.785, 39.600),
        "Freweyni": (14.033, 39.560),
        "Bizet": (14.360, 39.260)
    },
    "Southern Zone": {
        "Maychew": (12.784, 39.538),
        "Alamata": (12.417, 39.583),
        "Korem": (12.505, 39.523),
        "Mehoni": (12.800, 39.633)
    },
    "South Eastern Zone": {
        "Mekelle": (13.496, 39.475),
        "Samre": (13.183, 39.200),
        "Hagere Selam": (13.650, 39.167)
    },
    "Western Zone": {
        "Shire (Inda Selassie)": (14.102, 38.283),
        "Humera": (14.298, 36.618),
        "Sheraro": (14.385, 37.761),
        "Lugdi": (14.210, 36.560)
    }
}

# Administrative Sidebar Navigation
st.sidebar.header("📍 Catchment Location Selector")
selected_zone = st.sidebar.selectbox("Select Administrative Zone", list(WOREDA_DATABASE.keys()))
selected_town = st.sidebar.selectbox("Select Woreda Hub / Town", list(WOREDA_DATABASE[selected_zone].keys()))
lat, lon = WOREDA_DATABASE[selected_zone][selected_town]

st.sidebar.info(f"**Selected Coordinates:**\nLat: {lat:.3f}°N | Lon: {lon:.3f}°E")

# ==========================================
# 3. HISTORICAL DATA GENERATION (1998-2023)
# ==========================================
@st.cache_data
def load_historical_data(town_name):
    """Generates synthetic 26-year daily rainfall data for simulation."""
    dates = pd.date_range(start="1998-01-01", end="2023-12-31", freq="D")
    np.random.seed(abs(hash(town_name)) % 100000)
    
    # Simulate seasonal rainfall (Kiremt monsoon focused in July-August)
    doy = dates.dayofyear
    seasonal_intensity = np.exp(-0.5 * ((doy - 215) / 30) ** 2)
    daily_rain = np.random.exponential(scale=5.0, size=len(dates)) * seasonal_intensity
    daily_rain[daily_rain < 0.5] = 0.0  # Zero out light trace rainfall
    
    df = pd.DataFrame({
        "Date": dates,
        "Year": dates.year,
        "Month": dates.month_name(),
        "Daily_Precipitation_mm": np.round(daily_rain, 2)
    })
    return df

df_historical = load_historical_data(selected_town)

# ==========================================
# FEATURE 1: SINGLE-YEAR DAILY EXPORT
# ==========================================
st.subheader("🗓️ 1. Single-Year Daily Rainfall Query & Export")
st.caption("Select a single calendar year to inspect daily records and export raw CSV data.")

col1, col2 = st.columns([1, 2])

with col1:
    available_years = sorted(df_historical["Year"].unique(), reverse=True)
    selected_year = st.selectbox("Select Calendar Year", available_years)
    
    df_single_year = df_historical[df_historical["Year"] == selected_year].copy()
    
    total_annual = df_single_year["Daily_Precipitation_mm"].sum()
    max_daily = df_single_year["Daily_Precipitation_mm"].max()
    rain_days = (df_single_year["Daily_Precipitation_mm"] > 0.1).sum()
    
    st.metric(f"Total Rainfall ({selected_year})", f"{total_annual:.1f} mm")
    st.metric(f"Peak Daily Storm", f"{max_daily:.1f} mm")
    st.metric(f"Rainy Days", f"{rain_days} days")
    
    # CSV Export Button for Single Year
    csv_bytes = df_single_year.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Export {selected_year} Daily Data (CSV)",
        data=csv_bytes,
        file_name=f"{selected_town}_Daily_Rainfall_{selected_year}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    st.dataframe(
        df_single_year[["Date", "Month", "Daily_Precipitation_mm"]],
        use_container_width=True,
        height=300
    )

st.markdown("---")

# ==========================================
# FEATURE 2: 26-YEAR VISUALS & GATED EXPORT
# ==========================================
st.subheader("📊 2. 26-Year Historical Analysis & Multi-Year Distribution")
st.caption("Visualization of the full 26-year daily record and multi-year mean profile. Exporting underlying matrices requires clearance.")

# Compute annual totals and monthly mean profile
annual_totals = df_historical.groupby("Year")["Daily_Precipitation_mm"].sum()
monthly_means = df_historical.groupby("Month", sort=False)["Daily_Precipitation_mm"].mean()

# Render Matplotlib Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

# Plot 1: Annual Rain Totals
ax1.bar(annual_totals.index, annual_totals.values, color="#1D4ED8")
ax1.set_title(f"Annual Rainfall Totals (1998–2023) - {selected_town}", fontsize=11, fontweight="bold")
ax1.set_xlabel("Year")
ax1.set_ylabel("Total Precipitation (mm)")
ax1.grid(True, linestyle="--", alpha=0.5)

# Plot 2: Monthly Average Profile
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_agg = df_historical.groupby(df_historical["Date"].dt.month)["Daily_Precipitation_mm"].mean()
ax2.plot(months, monthly_agg.values, marker="o", color="#DC2626", linewidth=2)
ax2.set_title(f"Mean Monthly Distribution (26-Year Average)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Month")
ax2.set_ylabel("Mean Daily Rainfall (mm)")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
st.pyplot(fig)

# Password-Gated Export Section
st.markdown("### 🔒 Permission-Gated Multi-Year Export")
st.info("Visual plots above are viewable to all users. Full multi-year matrices and high-resolution vector figures require an administrative passcode.")

gate_col1, gate_col2 = st.columns([1, 1])

with gate_col1:
    access_code = st.text_input("Enter Clearance Passcode", type="password", help="Contact administrator for authorization.")

with gate_col2:
    st.write(" ") # Spacing offset
    st.write(" ")
    # Simple passcode check (Can be linked to st.secrets["ADMIN_PASSCODE"] in production)
    if access_code == "TigrayHydro2026":
        st.success("✅ Access Granted: Administrative Clearance Validated")
        
        # Prepare figure download buffer
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
        img_buf.seek(0)
        
        # Prepare full matrix CSV buffer
        matrix_csv = df_historical.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download High-Res Plot (PNG)",
            data=img_buf,
            file_name=f"{selected_town}_26Year_Analysis.png",
            mime="image/png"
        )
        st.download_button(
            label="📥 Download Full 26-Year Matrix (CSV)",
            data=matrix_csv,
            file_name=f"{selected_town}_Full_26Year_Matrix_1998_2023.csv",
            mime="text/csv"
        )
    elif access_code != "":
        st.error("❌ Invalid Passcode. Access Denied.")

st.markdown("---")

# ==========================================
# FEATURE 4: 5-DAY LIVE PRECIPITATION FORECAST
# ==========================================
st.subheader("🌦️ 3. Live 5-Day Rainfall Forecast (Open-Meteo Integration)")
st.caption("Retrieves short-term regional weather predictions via Open-Meteo live API.")

@st.cache_data(ttl=3600)  # Cache for 1 hour to optimize performance
def fetch_5day_forecast(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum,precipitation_probability_max",
        "timezone": "Africa/Addis_Ababa",
        "forecast_days": 5
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        df_fc = pd.DataFrame({
            "Date": daily.get("time", []),
            "Expected Precipitation (mm)": daily.get("precipitation_sum", []),
            "Rain Probability (%)": daily.get("precipitation_probability_max", [])
        })
        return df_fc, None
    except Exception as e:
        return None, str(e)

df_forecast, err = fetch_5day_forecast(lat, lon)

if err:
    st.error(f"Unable to fetch live forecast data: {err}")
else:
    fc_col1, fc_col2 = st.columns([1, 2])
    
    with fc_col1:
        st.write(f"**Location:** {selected_town}")
        st.dataframe(df_forecast, use_container_width=True, hide_index=True)
        
    with fc_col2:
        st.bar_chart(
            df_forecast.set_index("Date")["Expected Precipitation (mm)"],
            use_container_width=True
        )

# Footer
st.markdown("---")
st.caption("Developed by Tsegay Ayele Kidane | Water Resources Specialist & Hydraulic Engineer")
