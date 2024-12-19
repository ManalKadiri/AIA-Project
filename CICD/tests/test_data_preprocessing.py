import pandas as pd
import pytest
from datetime import datetime
import boto3
from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

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

# Téléchargement depuis S3
def download_file_from_s3():
    if not os.path.exists(local_file_path):  # Évite de télécharger plusieurs fois
        try:
            print("Downloading file from S3...")
            s3.download_file(bucket_name, object_key, local_file_path)
            print("Download completed:", local_file_path)
        except Exception as e:
            print("Error downloading file:", e)
            pytest.fail("Unable to download the file from S3")

# Test pour vérifier les colonnes de données
def test_data_columns():
    # Télécharger le fichier avant de commencer les tests
    download_file_from_s3()

    # Charger le fichier fraudTest.csv
    df = pd.read_csv(local_file_path).sample(100)

    # Recalculer la colonne 'age' comme dans le script d'entraînement
    if 'dob' in df.columns:  # Assurez-vous que 'dob' est bien dans les colonnes
        df['age'] = ((datetime.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)

    # Liste des colonnes requises
    required_columns = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long", "is_fraud"]

    # Vérifier les colonnes manquantes
    missing_columns = set(required_columns) - set(df.columns)
    assert not missing_columns, f"Colonnes manquantes : {missing_columns}"

# Test pour vérifier la transformation de la colonne 'gender'
def test_gender_transformation():
    # Télécharger le fichier avant de commencer les tests
    download_file_from_s3()

    # Charger le fichier fraudTest.csv
    df = pd.read_csv(local_file_path).sample(100)

    # Transformer la colonne 'gender' comme dans le script d'entraînement
    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)

    # Vérifier que la transformation a été correctement appliquée
    assert set(df['gender'].unique()).issubset({0, 1}), "La colonne 'gender' contient des valeurs non transformées."
