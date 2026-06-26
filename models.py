import numpy as np
from transformers import pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt


def load_nlp_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def train_dl_model():
    X_seq = np.random.uniform(5, 25, size=(500, 5))
    y_seq = np.mean(X_seq, axis=1) / 25.0
    dl_model = MLPRegressor(hidden_layer_sizes=(
        32, 16), max_iter=500, random_state=42)
    dl_model.fit(X_seq, y_seq)
    return dl_model


def train_ml_model():
    X_ml = np.random.rand(500, 3)
    X_ml[:, 0] = X_ml[:, 0] * 10000
    y_ml = (X_ml[:, 0] / 2000) + ((1.0 - X_ml[:, 1]) * 5) + (X_ml[:, 2] * 7)
    ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ml_model.fit(X_ml, y_ml)
    return ml_model


def generate_shap_plot(model, input_array, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_array)
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.summary_plot(shap_values, input_array,
                      feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Impact on Delay Prediction")
    plt.tight_layout()
    return fig
