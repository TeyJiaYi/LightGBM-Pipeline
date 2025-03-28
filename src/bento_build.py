# src/bento_build.py

import subprocess
import bentoml
import mlflow

# Ensure we are using the HTTP URI for MLflow (not file://)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def main():
    print("Importing champion alias => Bento store ...")
    # This imports the model from the MLflow registry alias "production"
    bentoml.mlflow.import_model("loan_risk_model", "models:/LoanRiskModel@production")
    
    print("Running 'bentoml build' ...")
    subprocess.run(["bentoml", "build"], check=True)
    
    print("Running 'bentoml containerize LoanRiskService:latest' ...")
    subprocess.run(["bentoml", "containerize", "LoanRiskService:latest"], check=True)
    
    print("Bento build & containerize complete. Docker image => loanriskservice:latest")

if __name__=="__main__":
    main()
