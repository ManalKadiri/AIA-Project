import os
import boto3
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

# Paths and environment setup
output_dir = "/tmp"
preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
local_csv_path = os.path.join(output_dir, "fraudTest.csv")

# AWS Credentials (via variables d'environnement)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'eu-west-3')  # Par défaut : us-east-1
S3_BUCKET_NAME = "bucketkadiri"  # Remplacez par le nom exact de votre bucket
S3_OBJECT_KEY = "datasets/fraudTest.csv"

# Fonction pour télécharger le fichier depuis S3
def download_from_s3(bucket_name, object_key, local_file_path):
    print(f"Téléchargement du fichier {object_key} depuis S3...")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    try:
        s3_client.download_file(bucket_name, object_key, local_file_path)
        print(f"Fichier téléchargé avec succès dans : {local_file_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement du fichier depuis S3 : {e}")
        exit(1)

# Télécharger le fichier fraudTest.csv depuis S3
if not os.path.exists(local_csv_path):  # Vérifie si le fichier n'est pas déjà téléchargé
    download_from_s3(S3_BUCKET_NAME, S3_OBJECT_KEY, local_csv_path)

# Lire le fichier CSV
df = pd.read_csv(local_csv_path)

# Preprocessing
columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                   'trans_num', 'dob', 'unix_time']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_columns_to_drop)

df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
df['age'] = ((datetime.now() - pd.to_datetime(df['dob'])) / pd.Timedelta(days=365.25)).astype(int)

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

preprocessor.fit(X_train)

with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)

# Models
if 0 in Y_train.value_counts() and 1 in Y_train.value_counts():
    scale_pos_weight = (Y_train.value_counts()[0] / Y_train.value_counts()[1])
else:
    scale_pos_weight = 1

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
}

# MLflow setup
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', "http://localhost:4000"))
mlflow.set_experiment("mlflow-AIA")

try:
    mlflow.list_experiments()
    print("Connexion réussie à MLflow.")
except Exception as e:
    print(f"Erreur de connexion à MLflow : {e}")
    exit(1)

# Training
for model_name, model in models.items():
    print(f"Entraînement du modèle : {model_name}")
    with mlflow.start_run(run_name=model_name):
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.named_steps['classifier'].fit(X_train, Y_train)

        # Metrics
        Y_pred = clf.predict(X_test)
        auc_roc = roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1]) if hasattr(clf, "predict_proba") else None
        print(f"{model_name}: AUC={auc_roc}")

        mlflow.log_metric("AUC", auc_roc)

print("Training terminé.")
