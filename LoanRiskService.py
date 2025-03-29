# loanriskservice.py

import bentoml
from bentoml.io import JSON
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model from MLflow and create a runner
model_ref = bentoml.mlflow.get("loan_risk_model:latest")
runner = model_ref.to_runner()

# Create BentoML service using new decorator-based API
svc = bentoml.Service(name="loanriskservice", runners=[runner])

from fastapi.middleware.cors import CORSMiddleware

svc.mount_asgi_app(CORSMiddleware(
    app=svc.asgi_app,
    allow_origins=["*"],  # Or your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
))


@svc.api(input=JSON(), output=JSON())
async def predict(input_data):
    """
    Expects input JSON of the form: {"features": [1.2, 3.4, ...]}.
    """
    features = np.array([input_data["features"]], dtype=float)
    result = await runner.async_run(features)
    return {"risk_score": float(result[0])}
