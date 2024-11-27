FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /home/app

# Copier uniquement les fichiers nécessaires pour optimiser la construction
COPY requirements.txt /home/app/requirements.txt
COPY train.py /home/app/train.py

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /home/app/requirements.txt
RUN pip install --no-cache-dir fsspec s3fs psycopg2-binary

# Installer MLflow et ses dépendances pour le tracking
RUN pip install --no-cache-dir mlflow[extras] boto3

# Configurer MLflow pour pointer vers votre serveur local existant
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Définir une variable d'environnement pour le port
ENV PORT=5000

# Lancer le script d'entraînement
CMD ["python", "train.py"]
