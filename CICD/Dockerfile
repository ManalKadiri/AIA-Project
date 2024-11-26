FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /home/app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y nano unzip curl && \
    rm -rf /var/lib/apt/lists/*  # Nettoyage des caches APT pour réduire la taille de l'image

# Installer Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Installer AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && \
    rm -rf awscliv2.zip aws  # Supprimer les fichiers temporaires

# Copier le fichier requirements.txt et installer les packages Python requis
COPY requirements.txt /dependencies/requirements.txt
RUN pip install --no-cache-dir -r /dependencies/requirements.txt

# Installer pytest séparément si nécessaire pour les tests (uniquement si pas déjà dans requirements.txt)
RUN pip install --no-cache-dir pytest

# Définir les variables d'environnement
ENV PORT=5000

# Copier le code source de l'application
COPY . /home/app

# Lancer le serveur MLflow avec les variables d'environnement sensibles injectées au runtime
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${ARTIFACT_STORE_URI} -p ${PORT}"]
