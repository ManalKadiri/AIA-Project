# Utiliser une image de base Python
FROM python:3.9-slim

# Installer les dépendances nécessaires
RUN pip install --upgrade pip
RUN pip install boto3 fastapi uvicorn openpyxl matplotlib pandas numpy plotly seaborn mlflow xgboost

# Définir les variables d'environnement dynamiques (elles seront injectées via Jenkins ou docker-compose)
ENV APP_URI_MODEL=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_DEFAULT_REGION=""

# Copier le code source dans le conteneur
COPY . /home/app

# Définir le répertoire de travail
WORKDIR /home/app

# Exposer le port (optionnel, utile pour des tests locaux)
EXPOSE 8000

# Commande pour démarrer l'application FastAPI en utilisant la variable d'environnement $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port=${PORT:-8000}"]
