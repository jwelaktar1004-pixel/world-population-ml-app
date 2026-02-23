import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("world_population.csv")

st.title("🌍 Global Population Forecast App")

st.write("Predict 2030 population using trained ML model.")

# Select country
country = st.selectbox("Select Country", df["Country/Territory"].unique())

# Get selected country data
country_data = df[df["Country/Territory"] == country]

# Prepare features (same as training)
year_index = 2030 - 1970
lag_population = np.log1p(country_data["2022 Population"].values[0])

continent_encoded = 0  # Optional: simplify if you didn't save encoder
area = country_data["Area (km²)"].values[0]

# Feature array
features = np.array([[lag_population, year_index, continent_encoded, area]])

# Predict
prediction_log = model.predict(features)
prediction = np.expm1(prediction_log)

st.subheader(f"Predicted 2030 Population for {country}:")
st.success(f"{int(prediction[0]):,}")