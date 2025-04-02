from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import boto3
import os
import yaml


# === Config ===
ROOT_DIR = os.getcwd()
DB_PATH = os.path.join(ROOT_DIR, "loan_predictions.db")
S3_BUCKET = "ml-predictions-loanrisk"
S3_PREFIX = "prediction"  # ✅ matches new structure
REGION = "us-east-1"
UPLOAD_CONFIG = os.path.join(ROOT_DIR, "config", "upload_config.yaml")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def save_upload_date(date_str):
    config = {"last_uploaded_date": date_str}
    os.makedirs(os.path.dirname(UPLOAD_CONFIG), exist_ok=True)
    with open(UPLOAD_CONFIG, "w") as f:
        yaml.dump(config, f)
    print(f"📌 Saved last upload date: {date_str} to {UPLOAD_CONFIG}")


def export_and_upload():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM loan_predictions
        WHERE timestamp >= datetime('now', '-1 day')
    """, conn)
    conn.close()

    if df.empty:
        print("⚠️ No prediction data found in the last 24 hours.")
        return

    export_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"predictions_{export_date}.csv"
    local_path = os.path.join(ROOT_DIR, csv_filename)
    df.to_csv(local_path, index=False)
    print(f"✅ Exported {len(df)} rows to {local_path}")

    # Upload to S3
    s3_key = f"{S3_PREFIX}/{export_date}/predictions.csv"
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(local_path, S3_BUCKET, s3_key)

    s3_url = f"https://s3.console.aws.amazon.com/s3/object/{S3_BUCKET}?prefix={s3_key}"
    print(f"✅ Uploaded to S3: {s3_url}")

    os.remove(local_path)
    print(f"🧹 Deleted local file {csv_filename}")

    # Save last uploaded date
    save_upload_date(export_date)


# === Define DAG ===
with DAG(
    "daily_upload_predictions_to_s3",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="Daily upload of prediction results to S3",
) as dag:

    upload_task = PythonOperator(
        task_id="export_and_upload_to_s3",
        python_callable=export_and_upload
    )
