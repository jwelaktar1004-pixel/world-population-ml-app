import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Global Population Forecast | Jwel Aktar",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# CUSTOM DARK STYLE
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #00FFAA;
    box-shadow: 0 0 8px #00FFAA;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("world_population.csv")

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("👨‍💻 Jwel Aktar")
st.sidebar.image("profile.jpg", width=180)

st.sidebar.markdown("""
**Global Population Forecasting App**  
Model: XGBoost Regression  
Year: 2030 Prediction
""")

st.sidebar.markdown("### 🔗 Connect With Me")
st.sidebar.markdown("[💼 LinkedIn](https://www.linkedin.com/in/jwel-aktar-61436b18a/)")
st.sidebar.markdown("[🌐 GitHub Repo](https://github.com/jwelaktar1004-pixel/world-population-ml-app)")

st.sidebar.markdown("---")

country1 = st.sidebar.selectbox("🌎 Select Country 1", sorted(df["Country/Territory"].unique()))
country2 = st.sidebar.selectbox("🌍 Select Country 2", sorted(df["Country/Territory"].unique()))

# =====================================================
# MAIN TITLE
# =====================================================
st.title("🌍 Global Population Forecast Dashboard")
st.markdown("### Machine Learning Powered | Developed by Jwel Aktar")

st.markdown("---")

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_population(country):
    country_data = df[df["Country/Territory"] == country]

    year_index = 2030 - 1970
    lag_population = np.log1p(country_data["2022 Population"].values[0])
    continent_encoded = 0
    area = country_data["Area (km²)"].values[0]

    features = np.array([[lag_population, year_index, continent_encoded, area]])

    prediction_log = model.predict(features)
    return int(np.expm1(prediction_log)[0])

pred1 = predict_population(country1)
pred2 = predict_population(country2)

# =====================================================
# ANIMATED KPI COUNTER
# =====================================================
def animated_counter(label, value):
    placeholder = st.empty()
    step = max(value // 60, 1)

    for i in range(0, value, step):
        placeholder.metric(label=label, value=f"{i:,}")
        time.sleep(0.005)

    placeholder.metric(label=label, value=f"{value:,}")

col1, col2 = st.columns(2)

with col1:
    animated_counter(f"📈 2030 Forecast - {country1}", pred1)

with col2:
    animated_counter(f"📈 2030 Forecast - {country2}", pred2)

st.markdown("---")

# =====================================================
# INTERACTIVE TREND COMPARISON
# =====================================================
st.subheader("📊 Historical Population Comparison")

years = [
    "1970 Population","1980 Population","1990 Population",
    "2000 Population","2010 Population",
    "2015 Population","2020 Population","2022 Population"
]

trend_df = pd.DataFrame({
    "Year": [int(y.split()[0]) for y in years],
    country1: df[df["Country/Territory"] == country1][years].values.flatten(),
    country2: df[df["Country/Territory"] == country2][years].values.flatten()
})

trend_long = pd.melt(
    trend_df,
    id_vars=["Year"],
    var_name="Country",
    value_name="Population"
)

fig = px.line(
    trend_long,
    x="Year",
    y="Population",
    color="Country",
    markers=True,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# WORLD MAP VISUALIZATION
# =====================================================
st.subheader("🗺 Global Population Map (2022)")

map_fig = px.choropleth(
    df,
    locations="Country/Territory",
    locationmode="country names",
    color="2022 Population",
    color_continuous_scale="Viridis",
    template="plotly_dark"
)

st.plotly_chart(map_fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("© 2026 Jwel Aktar | Advanced ML Deployment | Streamlit Cloud")