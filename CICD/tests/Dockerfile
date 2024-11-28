# Utiliser une image Python légère comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /home/tests

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y curl unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier les dépendances pour les tests
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers de tests
COPY . /home/tests

# Commande par défaut pour exécuter les tests
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
