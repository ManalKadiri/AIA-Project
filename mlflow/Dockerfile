FROM continuumio/miniconda3

WORKDIR /home/app

# Installer les dépendances nécessaires
RUN apt-get update && apt-get install -y nano unzip curl

# Installer Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Installer AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install

# Copier le fichier requirements.txt et installer les packages Python requis
COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

# Définir les variables d'environnement
ENV AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=BACKEND_STORE_URI
ENV ARTIFACT_STORE_URI=ARTIFACT_STORE_URI
ENV PORT=5000  

# Lancer le serveur MLflow
CMD echo "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}" && \
    echo "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}" && \
    echo "BACKEND_STORE_URI=${BACKEND_STORE_URI}" && \
    echo "ARTIFACT_STORE_URI=${ARTIFACT_STORE_URI}" && \
    mlflow server --host 0.0.0.0 --backend-store-uri "${BACKEND_STORE_URI}" --default-artifact-root "${ARTIFACT_STORE_URI}" -p ${PORT}
