# ad-click-predictor

📈 Ad Click-Through Rate (CTR) Predictor
This repository contains a pre-trained XGBoost machine learning model designed to predict the probability of a user clicking on a digital advertisement.

The model was trained on 1 million rows from the [Avazu CTR Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction) dataset from Kaggle.

Follow me on GitHub for more projects like this!

Model Performance
Model Type: XGBoost Classifier

Evaluation Metric: ROC-AUC

Performance: 0.7473 AUC on a 200,000-row test set.

🚀 How to Use This Model
You can run this entire project in just a few clicks using Google Colab.

Open the Colab Notebook:

Save Your Own Copy: In the Colab notebook, go to File > Save a copy in Drive. This will create your own editable version.

Run the Cells: Run the cells in the notebook from top to bottom. The notebook is already set up to:

Install the required libraries.

Download the model and encoder files from this repository.

Load them into memory.

Run a prediction on sample data to show you how it works.

🗂 Files in this Repository
xgb_ctr_model.joblib: The pre-trained XGBoost model file.

data_encoder.joblib: The OrdinalEncoder that was fitted on the training data. This is required to transform new data into the correct format for the model.

🙏 Credits
Dataset: Avazu Click-Through Rate Prediction on Kaggle.

Libraries: This project was built using Scikit-learn, XGBoost, and Pandas.
