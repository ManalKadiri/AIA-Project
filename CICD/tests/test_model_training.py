import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
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

# Download dataset from S3
try:
    print("Downloading file from S3...")
    s3.download_file(bucket_name, object_key, local_file_path)
    print("Download completed:", local_file_path)
except Exception as e:
    print("Error downloading file:", e)
    exit(1)


def test_model_accuracy():
    # Load dataset
    df = pd.read_csv(local_file_path)
    print("Data loaded:", df.head())


    # Prétraitement des données (repris de train.py)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
    df['age'] = ((pd.Timestamp.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)
    
    columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                       'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                       'trans_num', 'dob', 'unix_time']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]
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
    Recall = recall_score(Y_test, Y_pred)

    # Test pour vérifier la précision
    assert Recall > 0.7, f"Le rappel du modèle est insuffisante : {Recall}"
