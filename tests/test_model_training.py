import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import boto3

# Configurations pour AWS S3
BUCKET_NAME = "bucketkadiri"
S3_KEY = "datasets/fraudTest.csv"
LOCAL_PATH = "fraudTest.csv"

def download_from_s3(bucket_name, s3_key, local_path):
    """Télécharge un fichier depuis S3."""
    if not os.path.exists(local_path):
        try:
            print(f"Téléchargement du fichier {s3_key} depuis S3...")
            s3 = boto3.client("s3", region_name="eu-west-3")
            s3.download_file(bucket_name, s3_key, local_path)
            print(f"Fichier téléchargé avec succès : {local_path}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du téléchargement depuis S3 : {e}")

# Télécharger le fichier fraudTest.csv si nécessaire
download_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)

def test_model_accuracy():
    # Charger un échantillon de données pour les tests
    df = pd.read_csv(LOCAL_PATH).sample(1000, random_state=42)

    # Prétraitement des données
    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
    else:
        raise KeyError("La colonne 'gender' est absente du dataset.")

    if 'dob' in df.columns:
        df['age'] = ((pd.Timestamp.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)
    else:
        raise KeyError("La colonne 'dob' est absente du dataset.")
    
    columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                       'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                       'trans_num', 'dob', 'unix_time']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]
    if not all(col in df.columns for col in final_column_order):
        missing_columns = set(final_column_order) - set(df.columns)
        raise KeyError(f"Colonnes manquantes dans le dataset : {missing_columns}")

    X = df[final_column_order]
    Y = df['is_fraud']

    # Diviser les données en ensembles d'entraînement et de test
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

    # Entraîner un modèle simple (par exemple, Logistic Regression)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight="balanced"))
    ])
    model.fit(X_train, Y_train)

    # Prédictions et évaluation
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    # Test pour vérifier la précision
    assert accuracy > 0.7, f"La précision du modèle est insuffisante : {accuracy}"
    print(f"Test réussi ! Précision : {accuracy}")
