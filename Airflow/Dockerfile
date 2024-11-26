

FROM apache/airflow:2.6.3

# Copier les fichiers nécessaires
COPY requirements.txt /opt/requirements.txt

# Passer à l'utilisateur airflow pour installer les dépendances
USER airflow

# Installer les dépendances
RUN pip install --no-cache-dir -r /opt/requirements.txt
