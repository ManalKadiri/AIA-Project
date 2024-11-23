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
from dotenv import load_dotenv
import boto3

# Charger les variables d'environnement
load_dotenv()

# Configurations pour AWS S3
BUCKET_NAME = "bucketkadiri"  
S3_KEY = "datasets/fraudTest.csv"  # Chemin du fichier dans S3
LOCAL_PATH = "fraudTest.csv"  # Chemin local temporaire pour le fichier

# Fonction pour télécharger depuis S3
def download_from_s3(bucket_name, s3_key, local_path):
    """Télécharge un fichier depuis S3."""
    try:
        s3 = boto3.client("s3", region_name="eu-west-3")
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Fichier téléchargé depuis S3 : {local_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement depuis S3 : {e}")
        raise

# Vérifier si le fichier existe localement, sinon le télécharger depuis S3
if not os.path.exists(LOCAL_PATH):
    print(f"Téléchargement du fichier {LOCAL_PATH} depuis S3...")
    download_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)
    print("Téléchargement terminé.")

# Paths and environment setup
output_dir = "/tmp"
preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")

# Charger le dataset
try:
    df = pd.read_csv(LOCAL_PATH)
    print(f"Colonnes disponibles dans le dataset : {df.columns}")
except Exception as e:
    raise RuntimeError(f"Erreur lors de la lecture du fichier CSV : {e}")

# Prétraitement
if "gender" not in df.columns:
    raise KeyError("La colonne 'gender' est absente du dataset.")
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)  # Convertir `gender` en 1 (F) ou 0 (M)

# Calculer l'âge à partir de la date de naissance
if "dob" not in df.columns:
    raise KeyError("La colonne 'dob' est absente du dataset.")
df['age'] = ((datetime.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)

# Supprimer les colonnes inutiles
columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                   'trans_num', 'dob', 'unix_time']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Définir l'ordre final des colonnes
final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]

if not all(col in df.columns for col in final_column_order):
    raise KeyError(f"Colonne manquante dans le dataset : {set(final_column_order) - set(df.columns)}")

X = df[final_column_order]
Y = df['is_fraud']

# Séparation train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

# Préprocesseur
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

# Entraîner le préprocesseur sur les données d'entraînement
preprocessor.fit(X_train)

# Sauvegarder le préprocesseur
os.makedirs(output_dir, exist_ok=True)
with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)

# Modèles
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=(Y_train.value_counts()[0] / Y_train.value_counts()[1]))
}

# Configuration de MLflow
mlflow.set_tracking_uri(os.getenv("APP_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("mlflow-AIA")

# Entraînement des modèles
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Prétraiter les données
        transformed_X_train = preprocessor.transform(X_train)
        transformed_X_test = preprocessor.transform(X_test)

        # Entraîner le modèle
        clf.named_steps['classifier'].fit(transformed_X_train, Y_train)

        # Prédictions
        Y_pred = clf.named_steps['classifier'].predict(transformed_X_test)
        Y_proba = clf.named_steps['classifier'].predict_proba(transformed_X_test)[:, 1] if hasattr(clf.named_steps['classifier'], "predict_proba") else None

        # Calculer les métriques
        accuracy = clf.named_steps['classifier'].score(transformed_X_test, Y_test)
        auc_roc = roc_auc_score(Y_test, Y_proba) if Y_proba is not None else None
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        print(f"{model_name} Results: Accuracy={accuracy}, AUC ROC={auc_roc}, Recall={recall}, F1={f1}")
        mlflow.log_metric("Accuracy", accuracy)
        if auc_roc is not None:
            mlflow.log_metric("AUC ROC", auc_roc)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        # Sauvegarder le modèle et les artefacts
        mlflow.sklearn.log_model(clf, model_name)
        mlflow.log_artifact(preprocessor_path)

print("Entraînement terminé.")

# Supprimer le fichier local après utilisation
if os.path.exists(LOCAL_PATH):
    os.remove(LOCAL_PATH)
    print(f"Fichier temporaire supprimé : {LOCAL_PATH}")
