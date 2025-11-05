# ad-click-predictor

📈 Ad Click-Through Rate (CTR) Predictor
This repository contains a pre-trained XGBoost machine learning model designed to predict the probability of a user clicking on a digital advertisement.

The model was trained on 1 million rows from the Avazu CTR Prediction dataset from Kaggle.

Model Performance
Model Type: XGBoost Classifier

Evaluation Metric: ROC-AUC

Performance: 0.7473 AUC on a 200,000-row test set.

🗂 Files in this Repository
xgb_ctr_model.joblib: The pre-trained XGBoost model file.

data_encoder.joblib: The OrdinalEncoder that was fitted on the training data. This is required to transform new data into the correct format for the model.

🚀 How to Use This Model
You can load and use this model directly from this GitHub repository. Here is a complete Python code sample for a new project (e.g., in Google Colab).

Python

import joblib
import pandas as pd
import requests
from io import BytesIO
import xgboost as xgb # You must have xgboost, pandas, and scikit-learn installed

# --- 1. Define File URLs ---
# ⚠️ REPLACE THESE with your own "Raw" GitHub file URLs
model_url = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/xgb_ctr_model.joblib'
encoder_url = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data_encoder.joblib'

print("Downloading model files...")

# --- 2. Download and Load Model ---
m_r = requests.get(model_url)
# Check for successful download (status code 200)
if m_r.status_code == 200:
    model_file = BytesIO(m_r.content)
    loaded_model = joblib.load(model_file)
else:
    print(f"Error downloading model. Status code: {m_r.status_code}")

# --- 3. Download and Load Encoder ---
e_r = requests.get(encoder_url)
# Check for successful download
if e_r.status_code == 200:
    encoder_file = BytesIO(e_r.content)
    loaded_encoder = joblib.load(encoder_file)
else:
    print(f"Error downloading encoder. Status code: {e_r.status_code}")

print("✅ Model and encoder successfully loaded!")

# --- 4. Create New, Unseen Data for Prediction ---
# This data must have the same 23 columns as the original training data
# 'hour_of_day' and 'day_of_week' are our engineered time features.
new_impressions = pd.DataFrame({
    'C1': [1005, 1002],
    'banner_pos': [0, 1],
    'site_id': ['1fbe01fe', '85f751fd'],     # Values the encoder has seen
    'site_domain': ['f3845767', 'c4e18dd6'],
    'site_category': ['28905ebd', '50e219e0'],
    'app_id': ['ecad2386', 'a99f214a'],      # 'a99f214a' is a value it hasn't seen
    'app_domain': ['7801e8d9', '23450228'],
    'app_category': ['07d7df22', '0f2161f8'],
    'device_id': ['a99f214a', '0f2161f8'],
    'device_ip': ['ddd2926e', '382d5440'],
    'device_model': ['44956a24', 'SAMSUNG-SM-G900A'],
    'device_type': [1, 1],
    'device_conn_type': [0, 0],
    'C14': [15704, 22684],
    'C15': [320, 320],
    'C16': [50, 50],
    'C17': [1722, 2619],
    'C18': [0, 0],
    'C19': [35, 35],
    'C20': [-1, -1],
    'C21': [79, 43],
    'hour_of_day': [14, 18], # 2 PM and 6 PM
    'day_of_week': [3, 5]  # Wednesday and Friday
})

# --- 5. Preprocess the New Data ---
# Define the categorical columns (all except our engineered time features)
categorical_features = [col for col in new_impressions.columns if col not in ['hour_of_day', 'day_of_week']]

# Use the LOADED ENCODER to transform the categorical columns
new_impressions[categorical_features] = loaded_encoder.transform(new_impressions[categorical_features])

print("\nSuccessfully preprocessed new data.")

# --- 6. Make Predictions ---
# Use .predict_proba() to get probabilities
predictions_proba = loaded_model.predict_proba(new_impressions)

# Get the probability of a "click" (class 1)
click_probabilities = predictions_proba[:, 1]

print("\n--- Predictions ---")
for i, prob in enumerate(click_probabilities):
    print(f"Impression {i+1} Click Probability: {prob*100:.2f}%")

🙏 Credits
Dataset: Avazu Click-Through Rate Prediction on Kaggle.

Libraries: This project was built using Scikit-learn, XGBoost, and Pandas.
