import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import boto3

# Configurations pour AWS S3
BUCKET_NAME = "bucketkadiri"
S3_KEY = "datasets/fraudTest.csv"
LOCAL_PATH = "fraudTest.csv"

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

# Télécharger le fichier si nécessaire
if not os.path.exists(LOCAL_PATH):
    print(f"Téléchargement du fichier {LOCAL_PATH} depuis S3...")
    download_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)
    print("Téléchargement terminé.")

# Charger les données
try:
    df = pd.read_csv(LOCAL_PATH)
    print(f"Colonnes disponibles dans le dataset : {df.columns}")
except Exception as e:
    raise RuntimeError(f"Erreur lors de la lecture du fichier CSV : {e}")

# Prétraitement des données
if "gender" not in df.columns:
    raise KeyError("La colonne 'gender' est absente du dataset.")
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)  # Convertir `gender` en 1 (F) ou 0 (M)

if "dob" not in df.columns:
    raise KeyError("La colonne 'dob' est absente du dataset.")
df['age'] = ((pd.Timestamp.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)

columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                   'trans_num', 'dob', 'unix_time']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Vérification des colonnes finales
final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]

if not all(col in df.columns for col in final_column_order):
    raise KeyError(f"Colonne manquante dans le dataset : {set(final_column_order) - set(df.columns)}")

X = df[final_column_order]
Y = df['is_fraud']

# Diviser les données avec stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Vérifier si l'échantillon de test contient au moins deux classes
if len(Y_test.unique()) < 2:
    print("⚠️ L'échantillon de test contient une seule classe. Re-séparation des données avec un mélange aléatoire.")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=None, random_state=42
    )

# Appliquer SMOTE pour équilibrer les classes dans l'ensemble d'entraînement
smote = SMOTE(random_state=42)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)

print("Distribution après équilibrage (SMOTE) :")
print(Y_train_balanced.value_counts())

# Définir le modèle XGBoost
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(Y_train.value_counts()[0] / Y_train.value_counts()[1]),
    random_state=42
)

# Entraîner le modèle
print("Entraînement du modèle XGBoost...")
model.fit(X_train_balanced, Y_train_balanced)

# Prédictions
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:, 1]

# Calculer et valider AUC ROC
print("Calcul des métriques...")
try:
    auc_roc = roc_auc_score(Y_test, Y_prob)
    accuracy = accuracy_score(Y_test, Y_pred)

    # Validation des métriques uniquement basée sur l'AUC ROC
    assert auc_roc > 0.8, f"AUC ROC insuffisant : {auc_roc}"

    print(f"Validation réussie ! Métriques obtenues :")
    print(f"AUC ROC = {auc_roc}, Accuracy = {accuracy}")

except AssertionError as e:
    print(f"Erreur de validation : {e}")
    raise

# Supprimer le fichier local après utilisation
if os.path.exists(LOCAL_PATH):
    os.remove(LOCAL_PATH)
    print(f"Fichier temporaire supprimé : {LOCAL_PATH}")
