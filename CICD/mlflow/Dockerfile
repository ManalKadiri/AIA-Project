FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /home/app

# Installer les dépendances nécessaires
RUN apt-get update && apt-get install -y nano unzip curl

# Copier tous les fichiers du contexte de construction dans l'image
COPY . /home/app/

# Installer les dépendances Python
RUN pip install -r /home/app/requirements.txt
RUN pip install fsspec s3fs boto3 psycopg2-binary

# Définir une variable d'environnement non sensible
ENV PORT=5000

# Lancer le serveur MLflow en local
CMD mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /home/app/artifacts -p ${PORT}
