# Utiliser une image Python légère comme base
FROM python:3.9-slim

# Définir le répertoire de travail à /home
WORKDIR /home

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y curl unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier les dépendances pour les tests
COPY requirements.txt /home/
RUN pip install --no-cache-dir -r /home/requirements.txt

# Copier les fichiers de tests et le script `send_email.py`
COPY test_data_preprocessing.py /home/
COPY test_model_training.py /home/
COPY send_email.py /home/

# Commande par défaut pour exécuter les tests
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
