import bentoml
import pandas as pd
import numpy as np
from bentoml.io import JSON
import sqlite3
import uuid

# Load model
model_ref = bentoml.mlflow.get("loan_risk_model:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service("LoanRiskService", runners=[model_runner])




def save_prediction_to_db(input_data: dict, prediction: int):
    conn = sqlite3.connect("/tmp/loan_predictions.db")
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS loan_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            loan_id TEXT,
            payFrequency INTEGER,
            apr FLOAT,
            nPaidOff FLOAT,
            loanAmount FLOAT,
            originallyScheduledPaymentAmount FLOAT,
            leadType INTEGER,
            leadCost INTEGER,
            hasCF INTEGER,
            region_code INTEGER,
            prediction INTEGER
        )
    """)

    # Generate loan_id 
    loan_id = str(uuid.uuid4())

    values = (
        loan_id,
        input_data["payFrequency"],
        input_data["apr"],
        input_data.get("nPaidOff", None),
        input_data["loanAmount"],
        input_data["originallyScheduledPaymentAmount"],
        input_data["leadType"],
        input_data["leadCost"],
        input_data["hasCF"],
        input_data["region_code"],
        prediction
    )

    # Insert into table
    cursor.execute("""
        INSERT INTO loan_predictions (
            loan_id, payFrequency, apr, nPaidOff, loanAmount,
            originallyScheduledPaymentAmount, leadType, leadCost,
            hasCF, region_code, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, values)

    conn.commit()
    conn.close()



@svc.api(input=JSON(), output=JSON())
async def predict(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])

    # Cast to correct types as per MLflow schema
    cast_types = {
        "payFrequency": "int64",
        "apr": "float64",
        "nPaidOff": "float64",  # optional column
        "loanAmount": "float64",
        "originallyScheduledPaymentAmount": "float64",
        "leadType": "int64",
        "leadCost": "int64",
        "hasCF": "int64",
        "region_code": "int64"
    }

    df = df.astype(cast_types)

    result = await model_runner.async_run(df)

    prediction = int(result[0])
    save_prediction_to_db(input_data, prediction)

    risk = "Low Risk" if prediction == 1 else "High Risk"

    
    return {"risk": risk}
