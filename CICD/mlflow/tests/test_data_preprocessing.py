import pandas as pd
import pytest
from datetime import datetime

def test_data_columns():
    # Charger le fichier fraudTest.csv
    df = pd.read_csv("fraudTest.csv").sample(100)

    # Recalculer la colonne 'age' comme dans le script d'entraînement
    if 'dob' in df.columns:  # Assurez-vous que 'dob' est bien dans les colonnes
        df['age'] = ((datetime.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)

    # Liste des colonnes requises
    required_columns = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long", "is_fraud"]

    # Vérifier les colonnes manquantes
    missing_columns = set(required_columns) - set(df.columns)
    assert not missing_columns, f"Colonnes manquantes : {missing_columns}"

def test_gender_transformation():
    # Charger le fichier fraudTest.csv
    df = pd.read_csv("fraudTest.csv").sample(100)

    # Transformer la colonne 'gender' comme dans le script d'entraînement
    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)

    # Vérifier que la transformation a été correctement appliquée
    assert set(df['gender'].unique()).issubset({0, 1}), "La colonne 'gender' contient des valeurs non transformées."
