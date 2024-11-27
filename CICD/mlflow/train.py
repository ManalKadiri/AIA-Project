import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import boto3

# Charger les variables d'environnement
load_dotenv()

# Configurations pour AWS S3
BUCKET_NAME = "bucketkadiri"
S3_KEY = "datasets/fraudTest.csv"
LOCAL_PATH = "fraudTest.csv"

# Fonction pour télécharger le dataset depuis S3
def download_from_s3(bucket_name, s3_key, local_path):
    """Télécharge un fichier depuis S3."""
    try:
        s3 = boto3.client("s3", region_name="eu-west-3")
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Fichier téléchargé depuis S3 : {local_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement depuis S3 : {e}")
        raise

# Télécharger le dataset si nécessaire
if not os.path.exists(LOCAL_PATH):
    print(f"Téléchargement du fichier {LOCAL_PATH} depuis S3...")
    download_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)
    print("Téléchargement terminé.")

# Charger les données
df = pd.read_csv(LOCAL_PATH)

# Prétraitement
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)  # Convertir `gender` en 1 (F) ou 0 (M)
df['age'] = ((pd.Timestamp.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)

columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                   'trans_num', 'dob', 'unix_time']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Colonnes finales
final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]
X = df[final_column_order]
Y = df['is_fraud']

# Séparation train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

# Charger le préprocesseur (généré dans train.py)
preprocessor_path = "/tmp/preprocessor.pkl"
if not os.path.exists(preprocessor_path):
    raise FileNotFoundError(f"Préprocesseur introuvable : {preprocessor_path}")

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

# Définir les modèles à valider
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=(Y_train.value_counts()[0] / Y_train.value_counts()[1]))
}

# Validation des modèles
def validate_models(models, X_train, X_test, Y_train, Y_test):
    """Valide chaque modèle et vérifie les métriques."""
    validation_results = {}
    for model_name, model in models.items():
        print(f"Validation du modèle : {model_name}")
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

        # Afficher les résultats
        print(f"Résultats pour {model_name} :")
        print(f"Accuracy = {accuracy}, AUC ROC = {auc_roc}, Recall = {recall}, F1 Score = {f1}")

        # Vérifier les seuils
        if auc_roc is not None and auc_roc < 0.75:
            print(f"⚠️ AUC ROC trop faible pour {model_name}: {auc_roc}")
        if recall < 0.6:
            print(f"⚠️ Recall trop faible pour {model_name}: {recall}")

        # Enregistrer les résultats
        validation_results[model_name] = {
            "Accuracy": accuracy,
            "AUC ROC": auc_roc,
            "Recall": recall,
            "F1 Score": f1
        }

    return validation_results

# Lancer la validation
results = validate_models(models, preprocessor.transform(X_train), preprocessor.transform(X_test), Y_train, Y_test)
print("Validation terminée.")
print(results)

# Supprimer les fichiers temporaires
if os.path.exists(LOCAL_PATH):
    os.remove(LOCAL_PATH)
    print(f"Fichier temporaire supprimé : {LOCAL_PATH}")
