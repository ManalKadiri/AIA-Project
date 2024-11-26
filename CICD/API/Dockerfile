# Utiliser une image de base Python légère
FROM python:3.9-slim

# Mettre à jour pip et installer les dépendances nécessaires
RUN pip install --upgrade pip
RUN pip install boto3 fastapi uvicorn pandas numpy mlflow xgboost

# Copier le code source dans le conteneur
COPY . /home/app

# Définir le répertoire de travail
WORKDIR /home/app

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande pour démarrer l'application FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

