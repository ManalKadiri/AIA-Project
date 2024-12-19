import os
import shutil
import mlflow
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Obtenir les variables nécessaires
MLFLOW_TRACKING_URI = os.getenv("APP_URI_MODEL")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")  # Par défaut "default"
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/tmp/rollback_model")  # Dossier local pour le modèle

# Vérifier si MLFLOW_TRACKING_URI est défini
if not MLFLOW_TRACKING_URI:
    raise ValueError("La variable d'environnement MLFLOW_TRACKING_URI n'est pas définie.")

# Connecter au serveur MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Récupérer le dernier run stable
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    raise ValueError(f"L'expérience '{EXPERIMENT_NAME}' est introuvable.")

runs = client.search_runs(
    [experiment.experiment_id],
    filter_string="tags.action_required = 'stable'",  # Filtre pour trouver un modèle stable
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise ValueError("Aucun run stable trouvé pour l'expérience.")

stable_run = runs[0]

# Télécharger les artefacts du modèle
model_uri = f"runs:/{stable_run.info.run_id}/XGBoost"


if os.path.exists(LOCAL_MODEL_DIR):
    shutil.rmtree(LOCAL_MODEL_DIR)

# Utiliser mlflow.artifacts.download_artifacts
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=LOCAL_MODEL_DIR)

print(f"Retour au modèle stable effectué : run_id = {stable_run.info.run_id}")
print(f"Modèle téléchargé dans : {LOCAL_MODEL_DIR}")
