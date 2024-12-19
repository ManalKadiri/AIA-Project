import logging
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Combined Data Drift Detection")

# Paths (adjusted for Docker)
DATA_DIR = Path("/opt/airflow/dags")  # Matches volume from docker-compose.yaml
REPORTS_DIR = DATA_DIR / "reports"
DATA_FILE = DATA_DIR / "fraud_test.csv"

# Sample Size
SAMPLE_SIZE = 512  # Must be even

# Functions
def fetch_datasets(file_path: Path, sample_size: int):
    logger.info("Fetching datasets...")
    df = pd.read_csv(file_path)

    if len(df) < sample_size:
        raise ValueError(f"Not enough data in {file_path}. Found {len(df)} rows, need {sample_size}.")

    reference = df.sample(n=sample_size // 2, random_state=42)
    production = df.drop(reference.index).sample(n=sample_size // 2, random_state=42)

    # Simulate bias in `amt`
    production["amt"] *= 2

    logger.info("Datasets fetched and prepared.")
    return reference, production

def add_custom_message_to_html(report_path: Path, column_name: str):
    logger.info(f"Modifying the HTML report to add custom message: {report_path}")
    try:
        with open(report_path, "r") as file:
            soup = BeautifulSoup(file, "html.parser")

        # Create the custom message
        message = soup.new_tag("p")
        message.string = f"⚠️ La dérive dans le jeu de données est principalement causée par la colonne '{column_name}'."
        message["style"] = (
            "color: red; font-size: 24px; font-weight: bold; "
            "background-color: yellow; text-align: center; padding: 10px; border: 2px solid red;"
        )

        # Insert the message immediately after the <html> tag
        if soup.html:
            soup.html.insert(1, message)
            logger.info("Custom message successfully added at the very top of the HTML report.")
        else:
            logger.warning("Could not find <html> tag in the HTML report. Attempting to insert at the document start.")
            soup.insert(0, message)

        # Save the modified report
        with open(report_path, "w") as file:
            file.write(str(soup))
    except Exception as e:
        logger.error(f"Failed to modify the HTML report: {e}")

def generate_drift_report(reference, production, output_dir: Path):
    logger.info("Generating data drift report...")

    # Focus on `amt` drift
    amt_drift_report = Report(metrics=[ColumnDriftMetric(column_name="amt")])
    amt_drift_report.run(reference_data=reference, current_data=production)
    amt_output_file = output_dir / f"amt_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    amt_drift_report.save_html(str(amt_output_file))

    logger.info(f"Specific drift report for 'amt' saved to: {amt_output_file}")

    # Add custom message to the report
    add_custom_message_to_html(amt_output_file, "amt")

    # Verify the custom message was successfully added
    try:
        with open(amt_output_file, "r") as file:
            content = file.read()
            if "⚠️ La dérive dans le jeu de données est principalement causée par la colonne 'amt'." in content:
                logger.info("Custom message is present in the report.")
            else:
                logger.warning("Custom message is NOT present in the report.")
    except Exception as e:
        logger.error(f"Error verifying custom message in the report: {e}")

def run_data_drift_detection():
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reference, production = fetch_datasets(DATA_FILE, SAMPLE_SIZE)
        generate_drift_report(reference, production, REPORTS_DIR)
        logger.info("Data drift detection completed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

# Airflow DAG definition
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 1, 1),
}

with DAG(
    dag_id="data_drift_detection",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    detect_drift = PythonOperator(
        task_id="detect_data_drift",
        python_callable=run_data_drift_detection,
    )
