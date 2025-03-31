from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import boto3
import os

# === Config ===
DB_PATH = "D:\MLE\loan_predictions.db"
S3_BUCKET = "ml-predictions-loanrisk"
S3_PREFIX = "predictions"
REGION = "us-east-1"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

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
    local_path = os.path.join(os.getcwd(), csv_filename)
    df.to_csv(local_path, index=False)
    print(f"✅ Exported {len(df)} rows to {local_path}")

    s3_key = f"{S3_PREFIX}/{export_date}/predictions.csv"
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")

    os.remove(local_path)
    print(f"🧹 Deleted local file {csv_filename}")

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
