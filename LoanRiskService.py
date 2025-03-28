# loanriskservice.py

import bentoml
from bentoml.io import JSON
import numpy as np

# Load model from MLflow and create a runner
model_ref = bentoml.mlflow.get("loan_risk_model:latest")
runner = model_ref.to_runner()

# Create BentoML service using new decorator-based API
svc = bentoml.Service(name="loanriskservice", runners=[runner])

@svc.api(input=JSON(), output=JSON())
async def predict(input_data):
    """
    Expects input JSON of the form: {"features": [1.2, 3.4, ...]}.
    """
    features = np.array([input_data["features"]], dtype=float)
    result = await runner.async_run(features)
    return {"risk_score": float(result[0])}
