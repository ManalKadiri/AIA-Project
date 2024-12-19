import os
import mlflow
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Obtenir les variables nécessaires
MLFLOW_TRACKING_URI = os.getenv("APP_URI_MODEL")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")  # Par défaut "default"
PERFORMANCE_THRESHOLD = float(os.getenv("PERFORMANCE_THRESHOLD", 0.75))  # Par défaut 0.75

# Vérifier si MLFLOW_TRACKING_URI est défini
if not MLFLOW_TRACKING_URI:
    raise ValueError("La variable d'environnement MLFLOW_TRACKING_URI n'est pas définie.")

# Connecter au serveur MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Récupérer le dernier run
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    raise ValueError(f"L'expérience '{EXPERIMENT_NAME}' est introuvable.")

runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
if not runs:
    raise ValueError("Aucun run trouvé pour l'expérience.")

latest_run = runs[0]

# Obtenir la métrique de performance
Recall = latest_run.data.metrics.get("Recall", None)

# Logs pour diagnostiquer
print(f"Vérification des métriques du run ID: {latest_run.info.run_id}")
if Recall is None:
    print("Erreur : La métrique 'Recall' n'est pas trouvée dans le dernier run.")
    print("Suggestion : Vérifiez que la métrique est correctement loggée dans votre pipeline d'entraînement.")
    sys.exit(1)  # Échec de la validation
elif Recall < PERFORMANCE_THRESHOLD:
    print(f"Validation échouée : Recall ({Recall}) est en dessous du seuil ({PERFORMANCE_THRESHOLD}).")
    print("Suggestion : Envisagez un rollback vers un modèle précédent marqué 'stable'.")
    sys.exit(1)  # Échec de la validation
else:
    print(f"Validation réussie : Recall ({Recall}) atteint le seuil ({PERFORMANCE_THRESHOLD}).")
    sys.exit(0)  # Validation réussie
