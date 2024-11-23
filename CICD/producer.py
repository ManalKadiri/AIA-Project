
# Code inspired from Confluent Cloud official examples library
# https://github.com/confluentinc/examples/blob/7.1.1-post/clients/cloud/python/producer.py
# to get payment informations and then store them into kafka confluent cluster.

# The aim of this script is to request real time payment api :

from confluent_kafka import Producer
import json
import ccloud_lib # Library not installed with pip but imported from ccloud_lib.py
import numpy as np
import time
from datetime import datetime
import requests
import pandas as pd


# Initialize configurations from "python.config" file
CONF = ccloud_lib.read_ccloud_config("python.config")
TOPIC = "real_time_payment"

# Create Producer instance
producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
producer = Producer(producer_conf)

# Create topic if it doesn't already exist
ccloud_lib.create_topic(CONF, TOPIC)

# Paramaters to request API
url = "https://real-time-payments-api.herokuapp.com/current-transactions"
headers = {
    'accept': 'application/json'
}

# Request API and store data in a kafka confluent :

# Producer: Ajouter des vérifications pour garantir des données valides
try:
    while True:
        # Request the API
        response = requests.get(url, headers=headers)
        data_json = json.loads(response.json())["data"][0]

        record_key = "payment"

        # Extraire les données nécessaires avec validations
        try:
            amt = float(data_json[3])
            gender = str(data_json[6]).upper()
            lat = float(data_json[11])
            long = float(data_json[12])
            dob = pd.to_datetime(data_json[15], errors="coerce")
            merch_lat = float(data_json[17])
            merch_long = float(data_json[18])

            if pd.isnull(dob):
                raise ValueError("Invalid DOB format")

            # Préparation du message
            my_json = json.dumps({
                'amt': amt,
                'gender': gender,
                'lat': lat,
                'long': long,
                'dob': dob.strftime('%Y-%m-%d'),
                'merch_lat': merch_lat,
                'merch_long': merch_long
            })

            print(f"Producing record for transaction n°{data_json[16]}.")
            producer.produce(
                TOPIC,
                key=record_key,
                value=my_json,
            )

        except Exception as e:
            print(f"Skipping record due to invalid data: {e}")

        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    producer.flush()
