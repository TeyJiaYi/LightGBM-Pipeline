# src/bento_build.py

import subprocess
import bentoml

def main():
    # Re-import the champion model from alias=production
    print("Importing champion alias => Bento store ...")
    bentoml.mlflow.import_model("loan_risk_model", "models:/LoanRiskModel@production")

    print("Running 'bentoml build' ...")
    subprocess.run(["bentoml", "build"], check=True)

    print("Running 'bentoml containerize LoanRiskService:latest' ...")
    subprocess.run(["bentoml", "containerize", "LoanRiskService:latest"], check=True)


    print("Bento build & containerize complete. Docker image => loanriskservice:latest")

if __name__=="__main__":
    main()
