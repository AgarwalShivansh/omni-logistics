🚢 Omni-Logistics V2: Enterprise AI Supply Chain Engine

Omni-Logistics is a production-grade predictive engine designed to forecast cargo shipment delays by synthesizing heterogeneous data sources. By combining Natural Language Processing, Deep Learning (Sequential Modeling), and Core Ensemble Machine Learning, this application provides actionable supply chain intelligence.

🌐 Live Dashboard

Access the live application here:
https://omni-logistics-app.streamlit.app

🚀 Key Features

Multimodal AI Pipeline:
NLP Component: Fine-tuned Transformer (DistilBERT) to assess the risk impact of geopolitical news and weather reports.
Deep Learning Component: MLP Regressor to analyze temporal sequences of ship speed/trajectory health.
Core ML Component: RandomForestRegressor ensemble to synthesize distance, momentum, and risk scores into a final ETA delay prediction.
Geospatial Feature Engineering: Utilizes the Haversine formula to compute accurate great-circle distances between GPS coordinates.
Live Data Integration: Fetches real-time weather metrics using the Open-Meteo API.
Data Engineering: Persistent SQLite logging to track historical predictions for auditability.
Explainable AI: Integrated SHAP (SHapley Additive exPlanations) to visualize feature impact on model decisions.

🛠️ Tech Stack

Frontend/Dashboard: Streamlit
Machine Learning: Scikit-Learn, PyTorch, Hugging Face Transformers
Data Handling: Pandas, NumPy
Database: SQLite3
Visualization: Matplotlib, Plotly

🏗️ Architecture Overview

The application follows a modular design pattern to ensure scalability and maintainability:
app.py: The orchestrator—handles Streamlit UI and pipeline execution.
models.py: Encapsulates AI model training and inference logic.
features.py: Contains geospatial math and external API integration logic.
database.py: Manages SQLite connections and historical logging.

📈 Next Steps for Expansion

Deployment: Ready for containerization via Docker for cloud-native deployment.
MLOps: Implementing automated model retraining based on database logs to counter model drift.
GenAI Integration: Adding an LLM-based "Logistics Copilot" to query historical database trends in plain English.
