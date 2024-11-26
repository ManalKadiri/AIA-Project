import uvicorn
import numpy as np
import pandas as pd
import boto3
import pickle
import tempfile
import shutil
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

description = """
Welcome to the Fraud Detection API!\n
Provide the required information, and you will immediately know if the payment is fraudulent or not.

**Use the endpoint `/predict` to know if the payment is fraudulent or not!**
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Use this endpoint for getting predictions."
    }
]

app = FastAPI(
    title="💸 Fraud Detection API",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

# Input model for validation
class FraudDetectionInput(BaseModel):
    amt: float
    gender: str
    lat: float
    long: float
    dob: str
    merch_lat: float
    merch_long: float

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data to match the model's expected features.
    - Converts 'gender' to 1 (F) or 0 (M).
    - Converts 'dob' to 'age'.
    """
    try:
        # Convert 'gender' to binary (1 for 'F', 0 for 'M')
        data['gender'] = data['gender'].apply(lambda x: 1 if x == 'F' else 0)

        # Calculate 'age' from 'dob'
        data['age'] = ((datetime.now() - pd.to_datetime(data['dob'])) / pd.Timedelta(days=365.25)).astype(int)

        # Drop 'dob' as it is not used by the model
        data = data.drop(columns=['dob'])
        return data
    except Exception as e:
        logger.error("Error during preprocessing: %s", str(e))
        raise ValueError(f"Preprocessing error: {str(e)}")

@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Predict fraud for a transaction",
    description="This endpoint predicts whether a transaction is fraudulent based on input features.",
)
async def predict(
    data: List[FraudDetectionInput] = Body(
        ...,
        examples={
            "Prediction 0": {
                "summary": "Example of a non-fraudulent transaction",
                "description": "An example input for a non-fraudulent transaction.",
                "value": [
                    {
                        "amt": 2.86,
                        "gender": "M",
                        "lat": 33.9659,
                        "long": -80.9355,
                        "dob": "1968-03-19",
                        "merch_lat": 33.986391,
                        "merch_long": -81.200714
                    }
                ]
            },
            "Prediction 1": {
                "summary": "Example of a fraudulent transaction",
                "description": "An example input for a fraudulent transaction.",
                "value": [
                    {
                        "amt": 24.84,
                        "gender": "F",
                        "lat": 31.8599,
                        "long": -102.7413,
                        "dob": "1969-09-15",
                        "merch_lat": 32.575873,
                        "merch_long": -102.604290
                    }
                ]
            }
        }
    )
):
    try:
        # Convert input data to DataFrame
        fraud_features = pd.DataFrame([item.dict() for item in data])
        logger.info("Data received for prediction: %s", fraud_features)

        # Preprocess data
        fraud_features = preprocess_data(fraud_features)
        logger.info("Data after preprocessing: %s", fraud_features)

        # Configure S3 client and download model
        s3 = boto3.client('s3')
        bucket_name = "bucketkadiri"
        object_key = "Folder0/2/Model_RUN_ID/artifacts/XGBoost/model.pkl"
        local_file_path = "/home/app/model.pkl"
        
        # Check if the model already exists locally
        if not os.path.exists(local_file_path):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
            s3.download_file(bucket_name, object_key, temp_file_path)
            logger.info("Model successfully downloaded to a temporary file.")

            shutil.move(temp_file_path, local_file_path)
            logger.info("Model successfully moved to the final destination.")

        # Load model
        with open(local_file_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        logger.info("Model successfully loaded.")

        # Make predictions
        prediction = loaded_model.predict(fraud_features)
        response = {"prediction": prediction.tolist()}
        logger.info("Prediction completed successfully: %s", response)
        return response

    except ValueError as ve:
        logger.error("Preprocessing error: %s", str(ve))
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(ve)}")
    except Exception as e:
        logger.error("Error occurred during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
