import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from transformers import pipeline
from features import haversine_distance, fetch_live_weather_alert
from database import init_db, log_prediction, get_all_logs

init_db()
st.set_page_config(layout="wide", page_title="Omni-Logistics")

# CACHE BUSTER V6: Strict Mathematical Boundaries


@st.cache_resource
def load_ai_models_v6():
    # 1. NLP Pipeline
    nlp = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english")

    # 2. Deep Learning Model (Trajectory Health)
    # Train heavily on speeds from 0 to 25 so it perfectly understands "Stopped" vs "Fast"
    X_dl = np.random.uniform(0, 25, size=(5000, 5))
    # 25 knots = 1.0 Health, 0 knots = 0.0 Health
    y_dl = np.mean(X_dl, axis=1) / 25.0
    dl = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500,
                      random_state=42).fit(X_dl, y_dl)

    # 3. Core ML Engine (Delay Predictor)
    # Simulate 10,000 trips to ensure the AI knows exactly what to do at the extremes
    X_ml = np.random.rand(10000, 4)
    X_ml[:, 0] = X_ml[:, 0] * 20000  # Distance feature

    # THE RULES:
    # If Momentum is 1.0 (Max Speed), penalty is 0. If 0.0 (Stopped), penalty is 50 days.
    # If Risk is 0.0 (Clear Sky), penalty is 0. If 1.0 (Bad News), penalty is 30 days.
    momentum_penalty = (1.0 - X_ml[:, 2]) * 50.0
    risk_penalty = X_ml[:, 3] * 30.0
    # Small delay based on remaining distance
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
        ship_lat = st.number_input("Ship Latitude", value=35.0)
        ship_lon = st.number_input("Ship Longitude", value=-40.0)
        port_lat = st.number_input(
            "Port Latitude (e.g., NY is 40.71)", value=40.71)
        port_lon = st.number_input(
            "Port Longitude (e.g., NY is -74.00)", value=-74.00)
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
        st.header("🗺️ Live Tracking Map")
        map_data = pd.DataFrame(
            {'lat': [ship_lat, port_lat], 'lon': [ship_lon, port_lon]})
        st.map(map_data, zoom=3)

        st.header("🤖 AI Prediction & XAI")
        if st.button("Calculate ETA & Run SHAP Analysis", type="primary", use_container_width=True):

            # 1. Feature Engineering
            dist = haversine_distance(ship_lat, ship_lon, port_lat, port_lon)
            progress = 1.0 - \
                (dist / total_trip_dist) if total_trip_dist > 0 else 0.0

            # 2. NLP Pipeline
            nlp_result = nlp(news)[0]
            # Convert label to a 0.0 to 1.0 risk scale
            if nlp_result['label'] == 'NEGATIVE':
                risk = nlp_result['score']
            else:
                # If positive/neutral news, risk is mathematically flipped to be near 0
                risk = 1.0 - nlp_result['score']

            # 3. Deep Learning Sequence
            seq_input = np.array([[day1, day2, day3, day4, day5]])
            trajectory_health = dl.predict(seq_input)[0]
            # Force strictly between 0 and 1
            trajectory_health = np.clip(trajectory_health, 0.0, 1.0)

            # 4. Final Core ML
            features = np.array([[dist, progress, trajectory_health, risk]])
            delay = ml.predict(features)[0]

            log_prediction(ship_lat, ship_lon, port_lat, port_lon, news, delay)

            # Display Results
            st.subheader("Final Ensemble Output")
            if delay > 10.0:
                st.error(f"🚨 DISRUPTION: Predicted Delay is {delay:.1f} days!")
            elif delay > 3.0:
                st.warning(
                    f"⚠️ Moderate Delay: Predicted Delay is {delay:.1f} days.")
            else:
                st.success(
                    f"✅ On Track: Predicted Delay is {max(0, delay):.1f} days.")

            st.markdown("### 🧠 Explainable AI: Why did this happen?")

            # Custom, Clear Graph Generation
            feature_names = [
                "Distance (km)", "Trip Progress", "DL Momentum Health", "NLP Risk Score"]

            explainer = shap.TreeExplainer(ml)
            shap_vals = explainer.shap_values(features)[0]

            # Create a clean Matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 4))

            # Red for Delay Added (+), Green for Delay Subtracted/Saved (-)
            colors = ['#ff4d4d' if val > 0 else '#00cc66' for val in shap_vals]

            # Draw horizontal bars
            bars = ax.barh(feature_names, shap_vals, color=colors)
            ax.set_xlabel(
                "Days Added to (+) or Subtracted from (-) Base ETA", fontweight='bold')

            # Add a vertical line at 0
            ax.axvline(0, color='black', linewidth=1.5)

            # Add exact numbers next to the bars
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + (1.0 if width > 0 else -1.0)
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                        f"{width:+.1f} days",
                        va='center', ha='left' if width > 0 else 'right',
                        color='black', fontweight='bold')

            # Dynamically adjust the graph borders so text doesn't cut off
            max_bound = max(abs(shap_vals)) + 8
            ax.set_xlim(-max_bound, max_bound)

            st.pyplot(fig)

with tab2:
    st.header("🗄️ Database: Historical AI Logs")
    st.dataframe(get_all_logs(), use_container_width=True)
