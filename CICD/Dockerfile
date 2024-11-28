# Utiliser une image Python de base avec Miniconda
FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /home/app

# Installer les dépendances système
RUN apt-get update && apt-get install -y nano unzip curl

# Installer Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Installer AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install

# Copier le fichier requirements.txt et installer les packages nécessaires
COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

# Préparer les répertoires pour les artefacts et les fichiers temporaires
RUN mkdir -p /home/app/artifacts /home/app/tmp /home/app/mlruns /home/app/tests

# Copier les fichiers d'application, y compris train.py
COPY . /home/app
#COPY tests /home/app/tests
COPY test_data_preprocessing.py /home/app/test_data_preprocessing.py
COPY test_data_preprocessing.py /home/app/test_data_preprocessing.py
#COPY send_email.py /home/app/send_email.py

# Commande par défaut
CMD ["python", "train.py"]
