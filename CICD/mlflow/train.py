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
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import boto3
from dotenv import load_dotenv  # Charger les variables d'environnement

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# MLflow tracking URI (Heroku MLflow server)
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(mlflow_uri)

# Set the experiment name
mlflow.set_experiment("mlflow-AIA")

# AWS S3 Parameters
bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
object_key = os.getenv("OBJECT_KEY")
local_file_path = "fraudTest.csv"

# AWS Credentials (automatiquement pris en charge si configurés dans l'environnement)
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Download dataset from S3
try:
    print("Downloading file from S3...")
    s3.download_file(bucket_name, object_key, local_file_path)
    print("Download completed:", local_file_path)
except Exception as e:
    print("Error downloading file:", e)
    exit(1)

# Load dataset
df = pd.read_csv(local_file_path)
print("Data loaded:", df.head())

# Preprocessing
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
df['age'] = ((datetime.now() - pd.to_datetime(df['dob'])) / pd.Timedelta(days=365.25)).astype(int)
df = df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                      'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                      'trans_num', 'dob', 'unix_time'])

# Define column order for final DataFrame
final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]

X = df[final_column_order]
Y = df['is_fraud']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

# Preprocessor
numeric_features = ['amt', 'lat', 'long', 'age', 'merch_lat', 'merch_long']
categorical_features = ['gender']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', 'passthrough', categorical_features)
])

# Fit preprocessor on training data
preprocessor.fit(X_train)

# Save preprocessor
output_dir = "/tmp"
preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)

# Models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=(Y_train.value_counts()[0] / Y_train.value_counts()[1]))
}

# Training
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        # Train the model
        clf.fit(X_train, Y_train)

        # Predictions
        Y_pred = clf.predict(X_test)
        Y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        # Metrics
        accuracy = clf.score(X_test, Y_test)
        auc_roc = roc_auc_score(Y_test, Y_proba) if Y_proba is not None else None
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        print(f"{model_name} Results: Accuracy={accuracy}, AUC ROC={auc_roc}, Recall={recall}, F1={f1}")
        mlflow.log_metric("Accuracy", accuracy)
        if auc_roc is not None:
            mlflow.log_metric("AUC ROC", auc_roc)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        # Log model and artifacts
        mlflow.sklearn.log_model(clf, model_name)
        mlflow.log_artifact(preprocessor_path)

print("Training completed.")
