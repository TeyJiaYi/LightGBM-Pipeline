# loanriskservice.py

import bentoml
from bentoml.io import JSON
import numpy as np

# Retrieve the MLflow model from BentoML's model store
model_ref = bentoml.mlflow.get("loan_risk_model:latest")
LoanRiskRunner = model_ref.to_runner()

# Create the Bento service and attach the runner
svc = bentoml.Service("loanriskservice", runners=[LoanRiskRunner])

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    """
    Expects input JSON of the form: {"features": [1.2, 3.4, ...]}.
    """
    features = np.array([input_data["features"]], dtype=float)
    # Run inference on the model
    result = LoanRiskRunner.run(features)
    return {"risk_score": float(result[0])}
