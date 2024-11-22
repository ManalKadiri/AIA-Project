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
RUN pip install pytest
RUN pip install pytest requests


# Clé API de Postmark
ENV POSTMARK_API_KEY=${POSTMARK_API_KEY}

# Définir une variable d'environnement non sensible
ENV PORT=5000

# Copier le code source de l'application
COPY . /home/app

# Lancer le serveur MLflow avec les variables d'environnement sensibles injectées au runtime
CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "${BACKEND_STORE_URI}", "--default-artifact-root", "${ARTIFACT_STORE_URI}", "-p", "${PORT}"]
