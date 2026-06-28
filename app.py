import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from transformers import pipeline
from features import haversine_distance, fetch_live_weather_alert
from database import init_db, log_prediction, get_all_logs

# Constants for validation
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)


def validate_coords(lat, lon, label):
    if not (LAT_RANGE[0] <= lat <= LAT_RANGE[1] and LON_RANGE[0] <= lon <= LON_RANGE[1]):
        st.error(
            f"Invalid coordinates for {label}: Lat must be -90 to 90, Lon must be -180 to 180.")
        return False
    return True


# Initialize state for interactive map
if 'ship_lat' not in st.session_state:
    st.session_state.ship_lat = 35.0
if 'ship_lon' not in st.session_state:
    st.session_state.ship_lon = -40.0

init_db()
st.set_page_config(layout="wide", page_title="Omni-Logistics")


@st.cache_resource
def load_ai_models_v6():
    nlp = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english")
    X_dl = np.random.uniform(0, 25, size=(5000, 5))
    y_dl = np.mean(X_dl, axis=1) / 25.0
    dl = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500,
                      random_state=42).fit(X_dl, y_dl)
    X_ml = np.random.rand(10000, 4)
    X_ml[:, 0] = X_ml[:, 0] * 20000
    momentum_penalty = (1.0 - X_ml[:, 2]) * 50.0
    risk_penalty = X_ml[:, 3] * 30.0
    distance_factor = (X_ml[:, 0] * (1.0 - X_ml[:, 1])) / 5000.0
    y_ml = momentum_penalty + risk_penalty + distance_factor
    ml = RandomForestRegressor(
        n_estimators=50, random_state=42, min_samples_leaf=2).fit(X_ml, y_ml)
    return nlp, dl, ml


with st.spinner("Loading AI Models..."):
    nlp, dl, ml = load_ai_models_v6()

st.title("🚢 Omni-Logistics: AI Supply Chain Disruption Predictor")
tab1, tab2 = st.tabs(["Prediction Engine", "Historical Database Logs"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.header("📍 1. Geospatial & Trajectory")
        ship_lat = st.number_input(
            "Ship Latitude", value=st.session_state.ship_lat)
        ship_lon = st.number_input(
            "Ship Longitude", value=st.session_state.ship_lon)
        port_lat = st.number_input("Port Latitude", value=40.71)
        port_lon = st.number_input("Port Longitude", value=-74.00)
        total_trip_dist = st.number_input(
            "Total Planned Trip Distance (km)", value=10000.0)

        st.subheader("Deep Learning Input (Last 5 Days Speed)")
        day1 = st.slider("Day T-5 Speed (knots)", 0.0, 25.0, 25.0)
        day2 = st.slider("Day T-4 Speed (knots)", 0.0, 25.0, 25.0)
        day3 = st.slider("Day T-3 Speed (knots)", 0.0, 25.0, 25.0)
        day4 = st.slider("Day T-2 Speed (knots)", 0.0, 25.0, 25.0)
        day5 = st.slider("Day T-1 Speed (knots)", 0.0, 25.0, 25.0)

        st.header("📰 2. Live NLP Context")
        if st.button("🌐 Fetch Live Port Weather API"):
            st.session_state['news'] = fetch_live_weather_alert(
                port_lat, port_lon)
        news = st.text_area("Latest Weather/Geopolitical Alert:",
                            st.session_state.get('news', 'clear sky, perfect conditions'))

    with col2:
        st.header("🗺️ Live Tracking Map (Click to Update Ship Position)")
        m = folium.Map(location=[ship_lat, ship_lon], zoom_start=3)
        folium.Marker([ship_lat, ship_lon], popup="Ship",
                      icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker([port_lat, port_lon], popup="Port",
                      icon=folium.Icon(color="red")).add_to(m)

        map_data = st_folium(m, width=700, height=300)
        if map_data['last_clicked']:
            st.session_state.ship_lat = map_data['last_clicked']['lat']
            st.session_state.ship_lon = map_data['last_clicked']['lng']
            st.rerun()

        st.header("🤖 AI Prediction & XAI")
        if st.button("Calculate ETA & Run SHAP Analysis", type="primary", use_container_width=True):
            if validate_coords(ship_lat, ship_lon, "Ship") and validate_coords(port_lat, port_lon, "Port"):
                dist = haversine_distance(
                    ship_lat, ship_lon, port_lat, port_lon)
                progress = 1.0 - \
                    (dist / total_trip_dist) if total_trip_dist > 0 else 0.0
                nlp_result = nlp(news)[0]
                risk = nlp_result['score'] if nlp_result['label'] == 'NEGATIVE' else 1.0 - \
                    nlp_result['score']
                seq_input = np.array([[day1, day2, day3, day4, day5]])
                trajectory_health = np.clip(dl.predict(seq_input)[0], 0.0, 1.0)
                features = np.array(
                    [[dist, progress, trajectory_health, risk]])
                delay = ml.predict(features)[0]

                log_prediction(ship_lat, ship_lon, port_lat,
                               port_lon, news, delay)

                if delay > 10.0:
                    st.error(
                        f"🚨 DISRUPTION: Predicted Delay is {delay:.1f} days!")
                elif delay > 3.0:
                    st.warning(
                        f"⚠️ Moderate Delay: Predicted Delay is {delay:.1f} days.")
                else:
                    st.success(
                        f"✅ On Track: Predicted Delay is {max(0, delay):.1f} days.")

                st.markdown("### 🧠 Explainable AI")
                explainer = shap.TreeExplainer(ml)
                shap_vals = explainer.shap_values(features)[0]
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#ff4d4d' if val >
                          0 else '#00cc66' for val in shap_vals]
                bars = ax.barh(["Distance (km)", "Trip Progress",
                               "DL Momentum Health", "NLP Risk Score"], shap_vals, color=colors)
                ax.axvline(0, color='black', linewidth=1.5)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + (1.0 if width > 0 else -1.0), bar.get_y() + bar.get_height()/2,
                            f"{width:+.1f} days", va='center', ha='left' if width > 0 else 'right', fontweight='bold')
                st.pyplot(fig)

with tab2:
    st.header("🗄️ Database: Historical AI Logs")
    st.dataframe(get_all_logs(), use_container_width=True)
