import bentoml
import pandas as pd
import numpy as np
from bentoml.io import JSON

# Load model
model_ref = bentoml.mlflow.get("loan_risk_model:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service("LoanRiskService", runners=[model_runner])

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
    return {"risk_score": float(result[0])}
