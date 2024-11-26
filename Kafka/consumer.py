import json
import logging
import os
import pickle
from datetime import datetime

import boto3
import pandas as pd
from confluent_kafka import Consumer, Producer
import ccloud_lib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# AWS and Kafka Configurations
BUCKET_NAME = "bucketkadiri"
MODEL_PATH = "Folder0/2/RUN£_ID/artifacts/XGBoost/model.pkl"
PREPROCESSOR_PATH = "Folder0/2/RUN_ID/artifacts/preprocessor.pkl"

# Expected column names (directly defined here)
EXPECTED_COLUMNS = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]

# Functions
def initialize_aws_session():
    try:
        logger.info("Initializing AWS session...")
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        s3 = session.client("s3", region_name="eu-west-3")
        logger.info("AWS session initialized successfully.")
        return s3
    except Exception as e:
        logger.error(f"Error initializing AWS session: {e}")
        raise


def load_model_and_preprocessor_from_s3(s3):
    try:
        logger.info("Downloading and loading model and preprocessor from S3...")

        # Download and load the model
        model_obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_PATH)
        model = pickle.loads(model_obj['Body'].read())

        # Download and load the preprocessor
        preprocessor_obj = s3.get_object(Bucket=BUCKET_NAME, Key=PREPROCESSOR_PATH)
        preprocessor = pickle.loads(preprocessor_obj['Body'].read())

        logger.info("Model and preprocessor loaded successfully from S3.")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model or preprocessor from S3: {e}")
        raise


def send_email_notification(record):
    SENDER = "xxxx@gmail.com"
    RECIPIENT = "xxxxxx@gmail.com"
    SUBJECT = "Alert: Fraud Prediction Detected"
    BODY_TEXT = f"A fraud prediction was detected:\n\n{json.dumps(record, indent=4)}"

    try:
        logger.info("Sending email notification...")
        response = boto3.client("ses", region_name="eu-west-3").send_email(
            Destination={"ToAddresses": [RECIPIENT]},
            Message={
                "Body": {"Text": {"Charset": "UTF-8", "Data": BODY_TEXT}},
                "Subject": {"Charset": "UTF-8", "Data": SUBJECT},
            },
            Source=SENDER,
        )
        logger.info(f"Email sent successfully: {response['MessageId']}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def save_to_s3(s3, bucket_name, topic, key, data):
    """
    Enregistre les données sur S3 dans un dossier correspondant au topic Kafka.

    :param s3: Instance AWS S3
    :param bucket_name: Nom du bucket S3
    :param topic: Nom du topic Kafka (utilisé pour organiser les données dans S3)
    :param key: Clé (chemin relatif dans le bucket)
    :param data: Données à enregistrer
    """
    try:
        full_key = f"topics/real_time_payment_s3/{key}"  # Inclure le dossier 'topics' et le nom du topic
        logger.info(f"Saving record to S3 under key: {full_key}")
        s3.put_object(
            Bucket=bucket_name,
            Key=full_key,
            Body=json.dumps(data, indent=4),
            ContentType="application/json"
        )
        logger.info("Record successfully saved to S3.")
    except Exception as e:
        logger.error(f"Failed to save record to S3: {e}")



def preprocess_data(record, preprocessor):
    try:
        today = datetime.now()

        # Vérifier si 'dob' est présent et valide
        if 'dob' in record and record['dob']:
            try:
                dob = pd.to_datetime(record['dob'], errors='coerce')
                if pd.isnull(dob):  # Vérifier si la conversion a échoué
                    logger.warning(f"Invalid 'dob' format for record: {record['dob']}")
                    age = 0  # Valeur par défaut en cas d'erreur
                else:
                    # Calcul de l'âge
                    age = (today - dob).days // 365
            except Exception as e:
                logger.error(f"Error calculating age: {e}")
                age = 0  # Valeur par défaut en cas d'erreur
        else:
            logger.warning(f"'dob' is missing for record: {record}")
            age = 0  # Valeur par défaut si 'dob' est manquant

        # Convertir le genre en valeur numérique
        gender_value = 1 if record['gender'] == 'F' else 0

        # Créer un DataFrame pour la prédiction
        data_for_pred = pd.DataFrame([{
            "amt": record['amt'],
            "gender": gender_value,
            "lat": record['lat'],
            "long": record['long'],
            "age": age,
            "merch_lat": record['merch_lat'],
            "merch_long": record['merch_long']
        }])

        logger.info(f"DataFrame before preprocessing:\n{data_for_pred}")

        # Transformer les données avec le préprocesseur
        transformed_data = preprocessor.transform(data_for_pred)
        transformed_df = pd.DataFrame(transformed_data, columns=EXPECTED_COLUMNS)

        logger.info(f"Data transformed successfully. Shape: {transformed_df.shape}")
        return transformed_df

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise


if __name__ == "__main__":
    # Initialisation de la session AWS et téléchargement des fichiers nécessaires
    s3 = initialize_aws_session()
    model, preprocessor = load_model_and_preprocessor_from_s3(s3)

    try:
        # Initialisation du consumer et du producer Kafka
        logger.info("Initializing Kafka consumer and producer...")
        CONF = ccloud_lib.read_ccloud_config("python.config")
        consumer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
        consumer_conf["group.id"] = "payments_consumer_2"
        consumer_conf["auto.offset.reset"] = "earliest"

        producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
        producer = Producer(producer_conf)

        consumer = Consumer(consumer_conf)
        consumer.subscribe(["real_time_payment"])
        logger.info("Kafka consumer and producer initialized successfully.")

        # Injection d'une transaction frauduleuse pour le test
        test_fraudulent_transaction = {
            "amt": 9999.99,
            "gender": "F",
            "lat": 33.4418,
            "long": -94.0377,
            "dob": "1990-01-01",
            "merch_lat": 34.4418,
            "merch_long": -93.0377
        }
        logger.info("Injecting a fraud transaction for testing...")
        data_for_pred_norm = preprocess_data(test_fraudulent_transaction, preprocessor)
        prediction = 1  # Forcer la prédiction comme frauduleuse
        enriched_record = {
            **test_fraudulent_transaction,
            "age": (datetime.now() - pd.to_datetime(test_fraudulent_transaction["dob"])).days // 365,
            "prediction": prediction
        }
        send_email_notification(enriched_record)
        logger.info(f"Fraud transaction test executed: {enriched_record}")

        # Boucle principale du consumer Kafka
        logger.info("Starting Kafka consumer loop...")
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            elif msg.error():
                logger.warning(f"Consumer error: {msg.error()}")
                continue

            record_key = msg.key()
            record_value = msg.value()
            logger.info(f"Received record: key={record_key}, value={record_value}")

            try:
                # Convertir le message en dictionnaire Python
                record = json.loads(record_value)

                # Prétraiter les données
                data_for_pred_norm = preprocess_data(record, preprocessor)

                # Faire une prédiction
                prediction = model.predict(data_for_pred_norm)[0]

                # Enrichir l'enregistrement avec les données prétraitées et la prédiction
                enriched_record = {
                    "amt": record["amt"],
                    "gender": int(data_for_pred_norm["gender"].iloc[0]),
                    "lat": record["lat"],
                    "long": record["long"],
                    "merch_lat": record["merch_lat"],
                    "merch_long": record["merch_long"],
                    "age": int(data_for_pred_norm["age"].iloc[0]),
                    "prediction": int(prediction),
                }

                logger.info(f"Enriched record with prediction: {enriched_record}")

                try:
                    # Sauvegarder l'enregistrement enrichi sur S3
                    file_key = f"{datetime.now().isoformat()}_{record_key.decode('utf-8')}.json"
                    save_to_s3(s3, BUCKET_NAME, "real_time_payment_s3", file_key, enriched_record)

                    # Envoyer un email si une fraude est détectée
                    if prediction == 1:
                        send_email_notification(enriched_record)

    # Publier l'enregistrement enrichi dans le topic de sortie
                    producer.produce("real_time_payment_s3", key=str(record_key), value=json.dumps(enriched_record))
                    producer.flush()

                    # Committer le message
                    consumer.commit(asynchronous=False)

                except Exception as e:
                    logger.error(f"Error processing message: {e}")


                # Envoyer un email si une fraude est détectée
                if prediction == 1:
                    send_email_notification(enriched_record)

                # Publier l'enregistrement enrichi dans le topic de sortie
                producer.produce("real_time_payment_s3", key=str(record_key), value=json.dumps(enriched_record))
                producer.flush()

                # Committer le message
                consumer.commit(asynchronous=False)

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Consumer stopped manually.")
    finally:
        # Fermer les clients Kafka
        logger.info("Closing Kafka consumer and producer...")
        consumer.close()
        producer.flush()
        logger.info("Kafka consumer and producer closed.")