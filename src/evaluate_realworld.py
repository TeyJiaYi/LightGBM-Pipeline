# src/evaluate_realworld.py
import boto3
import os
import pandas as pd
import yaml
from datetime import datetime, timedelta
import json

# === Load config ===
with open("config/eval_realworld.yaml") as f:
    config = yaml.safe_load(f)

bucket = config["s3_bucket"]
prefix_pred = config["prediction_prefix"]
prefix_truth = config["truth_prefix"]
last_fetch_date = datetime.strptime(config["last_fetch_date"], "%Y-%m-%d")
today = datetime.today()

# === Ground truth labels ===
pos_labels = set(config["positive_loanstatus"])
neg_labels = set(config["negative_loanstatus"])
needed_cols = config["columns_needed"]

# === Local buffer ===
os.makedirs("tmp", exist_ok=True)
os.makedirs("data/retrain_buffer", exist_ok=True)
s3 = boto3.client("s3", region_name="us-east-1")

all_matches = []
total_eval_rows = 0
correct = 0
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

print(f"🔍 Evaluating from {last_fetch_date.date()} to {today.date()}")

current = last_fetch_date
while current <= today:
    ds = current.strftime("%Y-%m-%d")
    try:
        pred_key = f"{prefix_pred}{ds}/predictions.csv"
        truth_key = f"{prefix_truth}{ds}/loan_status.csv"

        pred_path = f"tmp/pred_{ds}.csv"
        truth_path = f"tmp/truth_{ds}.csv"

        s3.download_file(bucket, pred_key, pred_path)
        s3.download_file(bucket, truth_key, truth_path)

        pred_df = pd.read_csv(pred_path)
        truth_df = pd.read_csv(truth_path)[needed_cols]

        merged = pd.merge(pred_df, truth_df, on="loan_id", how="inner")
        merged = merged[merged["loanStatus"].isin(pos_labels.union(neg_labels))]

        merged["risk_indicator"] = merged["loanStatus"].apply(lambda x: 1 if x in pos_labels else 0)
        y_true = merged["risk_indicator"]
        y_pred = merged["prediction"]

        total_eval_rows += len(merged)
        correct += (y_true == y_pred).sum()

        # Save matched data to retrain_buffer
        merged.to_csv(f"data/retrain_buffer/validated_{ds}.csv", index=False)
        print(f"✅ Evaluated {ds}: {len(merged)} rows")

        all_matches.append(merged)

    except Exception as e:
        print(f"⚠️ Skipped {ds}: {e}")

    current += timedelta(days=1)

# === Metrics ===
if total_eval_rows == 0:
    print("❌ No evaluation data found")
    metrics = {"count": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}
else:
    df_all = pd.concat(all_matches)
    y_true = df_all["risk_indicator"]
    y_pred = df_all["prediction"]

    metrics = {
        "count": len(df_all),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "auc": round(roc_auc_score(y_true, y_pred), 4)
    }

print("📊 Final Evaluation Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Save
os.makedirs("metrics", exist_ok=True)
with open("metrics/realworld_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
