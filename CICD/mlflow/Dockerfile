FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /home/app

# Copier tous les fichiers du contexte de construction dans le répertoire de travail
COPY . /home/app/

# Installer les dépendances Python
RUN pip install -r /home/app/requirements.txt
RUN pip install fsspec s3fs
RUN pip install psycopg2-binary

# Définir une variable d'environnement non sensible
ENV PORT=5000

# Lancer le script d'entraînement
CMD ["python", "train.py"]
