import streamlit as st
import pandas as pd
import numpy as np
from features import haversine_distance, fetch_live_weather_alert
from models import load_nlp_pipeline, train_dl_model, train_ml_model, generate_shap_plot
from database import init_db, log_prediction, get_all_logs

init_db()
st.set_page_config(page_title="Omni-Logistics V2",
                   layout="wide", page_icon="🚢")


@st.cache_resource
def initialize_ai_engine():
    nlp = load_nlp_pipeline()
    dl = train_dl_model()
    ml = train_ml_model()
    return nlp, dl, ml


with st.spinner("Spinning up Omni-Logistics AI Engine..."):
    nlp_model, dl_trajectory_model, ml_final_model = initialize_ai_engine()

st.title("🚢 Omni-Logistics: AI Supply Chain Disruption Predictor")
st.markdown(
    "Predict cargo shipment delays using Geospatial Feature Engineering, Transformer NLP, Deep Sequence Modeling, Live API Integration, Explainable AI (SHAP), and SQLite Data Logging.")

tab1, tab2 = st.tabs(
    ["🔮 Live Prediction Engine", "📊 Historical Database Logs"])

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

        st.subheader("Deep Learning Input (Last 5 Days Speed)")
        day1 = st.slider("Day T-5 Speed (knots)", 0.0, 25.0, 18.0)
        day2 = st.slider("Day T-4 Speed (knots)", 0.0, 25.0, 17.5)
        day3 = st.slider("Day T-3 Speed (knots)", 0.0, 25.0, 12.0)
        day4 = st.slider("Day T-2 Speed (knots)", 0.0, 25.0, 8.0)
        day5 = st.slider("Day T-1 Speed (knots)", 0.0, 25.0, 5.0)

        st.header("📰 2. Live NLP Context")
        if st.button("🌐 Fetch Live Port Weather API"):
            with st.spinner("Calling Open-Meteo API..."):
                live_alert = fetch_live_weather_alert(port_lat, port_lon)
                st.session_state['news_text'] = live_alert

        news_input = st.text_area(
            "Latest Weather/Geopolitical Alert:",
            st.session_state.get(
                'news_text', "Massive dockworker strike expected."),
            height=80
        )

    with col2:
        st.header("🗺️ Live Tracking Map")
        map_data = pd.DataFrame({'lat': [ship_lat, port_lat], 'lon': [
                                ship_lon, port_lon], 'type': ['Ship', 'Port']})
        st.map(map_data, zoom=3, use_container_width=True)

        st.header("🤖 AI Prediction & XAI")
        if st.button("Calculate ETA & Run SHAP Analysis", type="primary", use_container_width=True):
            distance_km = haversine_distance(
                ship_lat, ship_lon, port_lat, port_lon)

            nlp_res = nlp_model(news_input)[0]
            risk_score = nlp_res['score'] if nlp_res['label'] == 'NEGATIVE' else (
                1.0 - nlp_res['score'])

            seq_input = np.array([[day1, day2, day3, day4, day5]])
            trajectory_health = dl_trajectory_model.predict(seq_input)[0]

            final_input = np.array(
                [[distance_km, trajectory_health, risk_score]])
            feature_names = [
                "Distance (km)", "DL Momentum Health", "NLP Risk Score"]
            predicted_delay = ml_final_model.predict(final_input)[0]

            log_prediction(ship_lat, ship_lon, port_lat,
                           port_lon, news_input, predicted_delay)

            st.subheader("Final Ensemble Output")
            if predicted_delay < 2.0:
                st.success(
                    f"✅ **On Track:** Predicted Delay is **{predicted_delay:.1f} days**.")
            else:
                st.error(
                    f"🚨 **DISRUPTION:** Predicted Delay is **{predicted_delay:.1f} days**!")

            st.markdown("### 🧠 Explainable AI (SHAP)")
            shap_fig = generate_shap_plot(
                ml_final_model, final_input, feature_names)
            st.pyplot(shap_fig)

with tab2:
    st.header("🗄️ Database: Historical AI Logs")
    if st.button("🔄 Refresh Logs"):
        pass
    logs_df = get_all_logs()
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True)
    else:
        st.info("No predictions logged yet. Run a calculation first!")
