import bentoml
from bentoml.io import JSON
import numpy as np

# import the champion from "models:/LoanRiskModel@production"
bentoml.mlflow.import_model("loan_risk_model", "models:/LoanRiskModel@production")

svc = bentoml.Service(name="LoanRiskService")

loan_model_ref = bentoml.mlflow.load_model("loan_risk_model:latest")

@svc.api(input=JSON(), output=JSON())
def predict(json_input):
    feats = np.array([json_input["features"]], dtype=float)
    if hasattr(loan_model_ref, "predict_proba"):
        prob = loan_model_ref.predict_proba(feats)[:,1]
    else:
        prob = loan_model_ref.predict(feats)
    return {"risk_score": float(prob[0])}
