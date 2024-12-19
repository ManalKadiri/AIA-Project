# Utiliser une image de base Python
FROM python:3.9-slim

# Installer les dépendances nécessaires
RUN pip install --upgrade pip
RUN pip install boto3
RUN pip install fastapi uvicorn openpyxl matplotlib pandas numpy plotly seaborn mlflow
RUN pip install xgboost

# Définir l'URI de suivi MLflow en tant que variable d'environnement
ENV MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI

# Définir les variables d'environnement AWS (remplacez les valeurs par vos propres credentials)
ENV AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=AWS_DEFAULT_REGION

# Copier le code source dans le conteneur
COPY . /home/app

# Définir le répertoire de travail
WORKDIR /home/app

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande pour démarrer l'application FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

