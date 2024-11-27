import os
import pickle
import boto3
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Configuration des chemins et S3
output_dir = "./artifacts"
preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
local_dataset_path = "./datasets/fraudTest.csv"
s3_bucket_name = "bucketkadiri"
s3_dataset_key = "datasets/fraudTest.csv"

# Téléchargement du fichier CSV depuis S3
def download_from_s3(bucket_name, key, local_path):
    print(f"Téléchargement du fichier depuis S3 : s3://{bucket_name}/{key}")
    s3 = boto3.client("s3", 
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    s3.download_file(bucket_name, key, local_path)
    print(f"Fichier téléchargé localement à : {local_path}")

# Récupérer le dataset
if not os.path.exists(local_dataset_path):
    download_from_s3(s3_bucket_name, s3_dataset_key, local_dataset_path)

# Charger le dataset
df = pd.read_csv(local_dataset_path)

# Prétraitement des données
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)  # Convertir `gender` en 1 (F) ou 0 (M)
df['age'] = ((datetime.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)

columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                   'trans_num', 'dob', 'unix_time']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Colonnes finales
final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]
X = df[final_column_order]
Y = df['is_fraud']

# Séparation des données
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

# Création du préprocesseur
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

# Ajustement du préprocesseur sur les données d'entraînement
preprocessor.fit(X_train)

# Sauvegarde du préprocesseur
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)
print(f"Préprocesseur sauvegardé à : {preprocessor_path}")

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=(Y_train.value_counts()[0] / Y_train.value_counts()[1]))
}

# Configuration MLflow pour usage local avec port 4000
mlflow.set_tracking_uri("http://mlflow-server:4000")
mlflow.set_experiment("mlflow-local")

# Validation des modèles
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        # Entraîner le modèle
        clf.named_steps['classifier'].fit(X_train, Y_train)

        # Prédictions
        Y_pred = clf.named_steps['classifier'].predict(X_test)
        Y_proba = clf.named_steps['classifier'].predict_proba(X_test)[:, 1] if hasattr(clf.named_steps['classifier'], "predict_proba") else None

        # Calcul des métriques
        accuracy = accuracy_score(Y_test, Y_pred)
        auc_roc = roc_auc_score(Y_test, Y_proba) if Y_proba is not None else None
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        print(f"{model_name} Results: Accuracy={accuracy}, AUC ROC={auc_roc}, Recall={recall}, F1={f1}")

        # Enregistrer les métriques dans MLflow
        mlflow.log_metric("Accuracy", accuracy)
        if auc_roc is not None:
            mlflow.log_metric("AUC ROC", auc_roc)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        # Sauvegarder le modèle et les artefacts dans MLflow
        mlflow.sklearn.log_model(clf, model_name)
        mlflow.log_artifact(preprocessor_path)

print("Entraînement terminé.")
