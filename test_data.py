import os
import pandas as pd
import boto3
from datetime import datetime
import uuid

# === CONFIG ===
S3_BUCKET = "ml-predictions-loanrisk"
REGION = "us-east-1"
PREDICTION_PREFIX = "prediction"
HISTORY_PREFIX = "history"

# === Generate Fake Data for Today ===
today = datetime.today().strftime("%Y-%m-%d")

# Generate 100 fake predictions
loan_ids = [str(uuid.uuid4()) for _ in range(100)]
predictions = pd.DataFrame({
    "loan_id": loan_ids,
    "payFrequency": [0] * 100,
    "apr": [0.15] * 100,
    "nPaidOff": [0.5] * 100,
    "loanAmount": [2000] * 100,
    "originallyScheduledPaymentAmount": [100] * 100,
    "leadType": [1] * 100,
    "leadCost": [20] * 100,
    "hasCF": [1] * 100,
    "region_code": [2] * 100,
    "prediction": [1 if i % 2 == 0 else 0 for i in range(100)]
})

# Generate ground truth for same loan IDs
statuses = [
    "Paid Off Loan", "Charged Off", "Returned Item", "External Collection",
    "Internal Collection", "Settlement Paid Off", "Settled Bankruptcy", "Charged Off Paid Off"
]
truth = pd.DataFrame({
    "loan_id": loan_ids,
    "loanStatus": [statuses[i % len(statuses)] for i in range(100)]
})

# === Save locally
os.makedirs("tmp_test_data", exist_ok=True)
pred_path = os.path.join("tmp_test_data", "predictions.csv")
truth_path = os.path.join("tmp_test_data", "loan_status.csv")
predictions.to_csv(pred_path, index=False)
truth.to_csv(truth_path, index=False)

# === Upload to S3
s3 = boto3.client("s3", region_name=REGION)

pred_key = f"{PREDICTION_PREFIX}/{today}/predictions.csv"
truth_key = f"{HISTORY_PREFIX}/{today}/loan_status.csv"

try:
    s3.upload_file(pred_path, S3_BUCKET, pred_key)
    s3.upload_file(truth_path, S3_BUCKET, truth_key)
    print(f"✅ Uploaded prediction to s3://{S3_BUCKET}/{pred_key}")
    print(f"✅ Uploaded ground truth to s3://{S3_BUCKET}/{truth_key}")
except Exception as e:
    print(f"❌ Upload failed: {e}")

# === Clean up
os.remove(pred_path)
os.remove(truth_path)
os.rmdir("tmp_test_data")
