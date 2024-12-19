import os
import pandas as pd
import numpy as np
import mlflow
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("mlflow-AIA")

# AWS S3 Setup
bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
object_key = os.getenv("OBJECT_KEY")
local_file_path = "fraudTest.csv"

s3 = boto3.client('s3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"))

# Download dataset
try:
    print("Downloading dataset from S3...")
    s3.download_file(bucket_name, object_key, local_file_path)
    print("Dataset downloaded successfully.")
except Exception as e:
    print("Error downloading file:", e)
    exit(1)

# Data Preparation
df = pd.read_csv(local_file_path)
print("Initial Data Loaded:", df.head())

df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
df['age'] = ((datetime.now() - pd.to_datetime(df['dob'])) / pd.Timedelta(days=365.25)).astype(int)
df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                 'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                 'trans_num', 'dob', 'unix_time'], inplace=True)

X = df[["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]]
Y = df['is_fraud']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), ["amt", "lat", "long", "age", "merch_lat", "merch_long"]),
    ('cat', 'passthrough', ['gender'])
])

# Models Definition
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=Y_train.value_counts()[0] / Y_train.value_counts()[1])
}

# Training and Logging for Each Model
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}...")

        clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        Y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        # Metrics Calculation
        metrics = {
            "Recall": recall_score(Y_test, Y_pred),
            "Precision":precision_score(Y_test, Y_pred),
            "F1 Score": f1_score(Y_test, Y_pred)
        }

        print(f"{model_name} Results: {metrics}")

        # Log metrics to MLflow
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # Log the model
        mlflow.sklearn.log_model(clf, model_name)
        print(f"{model_name} logged successfully.")

        # Register the model in the Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
        mlflow.register_model(model_uri=model_uri, name="fraud-detection-model")
        print(f"Model {model_name} registered successfully.")

print("Training completed.")
