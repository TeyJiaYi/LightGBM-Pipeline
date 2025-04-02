# dags/weekly_model_evaluation.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import yaml
import os
import shutil
import glob
import pandas as pd



# === Config ===
EVAL_CONFIG_PATH = "config/eval_realworld.yaml"
BUFFER_DIR = "data/retrain_buffer"
NEW_DATA_BASE = "data/raw/new"


def load_config():
    with open(EVAL_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_last_fetch_date(date_str):
    config = load_config()
    config["last_fetch_date"] = date_str
    with open(EVAL_CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

def update_eval_config(**context):
    today = datetime.today().strftime("%Y-%m-%d")

    config = {
        "s3_bucket": "ml-predictions-loanrisk",
        "last_fetch_date": "2020-01-01",  # default for first run
        "prediction_prefix": "prediction/",
        "truth_prefix": "history/",
        "prometheus_push_url": "http://localhost:5001/metrics",
        "columns_needed": ["loan_id", "loanStatus"],
        "positive_loanstatus": ["Paid Off Loan"],
        "negative_loanstatus": [
            "External Collection", "Internal Collection", "Returned Item",
            "Settlement Paid Off", "Settled Bankruptcy", "Charged Off", "Charged Off Paid Off"
        ],
        "retrain_threshold_metric": "accuracy",
        "retrain_threshold_value": 0.85,
        "retrain_min_rows": 1000,
        "min_metric_rows": 100
    }

    os.makedirs("config", exist_ok=True)
    if not os.path.exists(EVAL_CONFIG_PATH):
        with open(EVAL_CONFIG_PATH, "w") as f:
            yaml.dump(config, f)
        print(f"✅ Initialized evaluation config with default last_fetch_date")
    else:
        print(f"✅ Evaluation config already exists, skipping init")


def run_evaluation():
    result = subprocess.run(["python", "src/evaluate_realworld.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Evaluation script failed: {result.stderr}")


def total_buffered_rows():
    files = glob.glob(f"{BUFFER_DIR}/*.csv")
    total = sum(len(pd.read_csv(f)) for f in files)
    return total


def conditional_retrain():
    config = load_config()
    metric_name = config.get("retrain_threshold_metric", "accuracy")
    threshold_value = config.get("retrain_threshold_value", 0.85)
    min_rows = config.get("retrain_min_rows", 1000)
    min_metric_rows = config.get("min_metric_rows", 100)

    with open("metrics/realworld_metrics.json") as f:
        metrics = yaml.safe_load(f)

    metric_value = metrics.get(metric_name, 0)
    metric_sample_size = metrics.get("count", 0)
    row_count = total_buffered_rows()

    print(f"📊 Evaluation summary: {metric_name}={metric_value}, sample_size={metric_sample_size}, buffer_rows={row_count}")

    should_retrain = False

    if metric_value < threshold_value and metric_sample_size >= min_metric_rows:
        print(f"⚠️ Metric below threshold with sufficient samples")
        should_retrain = True

    elif row_count >= min_rows:
        print(f"⚠️ Buffered rows exceed threshold")
        should_retrain = True

    if should_retrain:
        today = datetime.today().strftime("%Y-%m-%d")
        new_data_path = os.path.join(NEW_DATA_BASE, today)
        os.makedirs(new_data_path, exist_ok=True)

        for file in glob.glob(f"{BUFFER_DIR}/*.csv"):
            base = os.path.basename(file)
            shutil.move(file, os.path.join(new_data_path, base))

        print(f"📦 Moved validated files to {new_data_path}")

        subprocess.run(["dvc", "repro", "preprocess", "train", "bento_build", "k8s_deploy"])

        shutil.rmtree(BUFFER_DIR, ignore_errors=True)
        os.makedirs(BUFFER_DIR, exist_ok=True)
        save_last_fetch_date(today)
        print(f"🧹 Buffer cleared and last_fetch_date updated to {today}.")
    else:
        print(f"✅ Retrain skipped based on thresholds")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10)
}

with DAG("weekly_model_evaluation",
         default_args=default_args,
         schedule_interval="@weekly",
         catchup=False,
         description="Evaluate production predictions vs ground truth") as dag:

    t1 = PythonOperator(
        task_id="update_eval_config",
        python_callable=update_eval_config
    )

    t2 = PythonOperator(
        task_id="run_weekly_evaluation",
        python_callable=run_evaluation
    )

    t3 = PythonOperator(
        task_id="conditional_retrain_if_needed",
        python_callable=conditional_retrain
    )

    t1 >> t2 >> t3