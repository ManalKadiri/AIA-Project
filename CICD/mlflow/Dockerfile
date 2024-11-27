FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /home/app

# Installer les dépendances nécessaires
RUN apt-get update && apt-get install -y nano unzip curl

# Installer Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Installer AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install

# Copier tous les fichiers du contexte de construction dans l'image
COPY . /home/app/

# Installer les dépendances Python
RUN pip install -r /home/app/requirements.txt
RUN pip install fsspec s3fs
RUN pip install psycopg2-binary


# Définir une variable d'environnement non sensible
ENV PORT=5000

# Lancer le serveur MLflow avec les variables d'environnement sensibles injectées au runtime
CMD mlflow server --host 0.0.0.0 --backend-store-uri "${BACKEND_STORE_URI}" --default-artifact-root "${ARTIFACT_STORE_URI}" -p ${PORT}
