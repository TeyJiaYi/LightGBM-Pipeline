import os
import sys
import yaml
import glob
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
import warnings

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LoanRisk")  
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")


def load_latest_model(models_dir):
    """
    Find the newest model file in `models_dir` with a known extension
    (e.g. .txt for LightGBM booster, or .pkl if it's a scikit-LGBM).
    Returns (model, model_path) if found, else raises error.
    """
    # might have .txt (LightGBM booster) or .pkl (pickled model).
    # Extend this pattern if store  model differently.
    pattern_txt = os.path.join(models_dir, "*.txt")
    pattern_pkl = os.path.join(models_dir, "*.pkl")

    model_files = glob.glob(pattern_txt) + glob.glob(pattern_pkl)
    if not model_files:
        raise FileNotFoundError(f"No model files found in: {models_dir}")

    # pick newest by modification time
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"Found latest model: {latest_model_path}")

    # If .txt, it's a native lightgbm booster
    if latest_model_path.endswith(".txt"):
        model = lgb.Booster(model_file=latest_model_path)
        return model, latest_model_path

    # If .pkl, scikit-learn or LGBMClassifier
    elif latest_model_path.endswith(".pkl"):
        import pickle
        with open(latest_model_path, "rb") as f:
            model = pickle.load(f)
        return model, latest_model_path
    
    else:
        raise ValueError(f"Unsupported model file: {latest_model_path}")

def main(config_path):
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. The stable final test set
    test_data_path = config.get("test_data_path", "data/test/test_loan.csv")
    if not os.path.isfile(test_data_path):
        raise FileNotFoundError(f"Test set not found: {test_data_path}")

    df_test = pd.read_csv(test_data_path)
    target_col = config.get("target_col", "risk_indicator")
    if target_col not in df_test.columns:
        raise ValueError(f"Target column '{target_col}' not in test data.")

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # 3. Load the newest model from models/
    models_dir = config.get("models_dir", "models")
    model, model_path = load_latest_model(models_dir)

    # 4. Generate predictions
    # If it's an lgb.Booster, we call predict(X_test).
    # If it's an sklearn model, we might call predict_proba or predict, etc.
    # We'll detect which type we have:
    if isinstance(model, lgb.Booster):
        # LightGBM booster => predict returns probabilities by default
        y_pred_prob = model.predict(X_test)
    else:
        # scikit-LGBM or similar => use predict_proba
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            # fallback: directly predict
            y_pred_prob = model.predict(X_test)
            # If it's not probabilities, won't be able to do AUC properly.

    # 5. Evaluate
    threshold = config.get("threshold", 0.5)
    y_pred_binary = (y_pred_prob > threshold).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    print("--- Final Evaluation (Test Set) ---")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # 6. Optionally log to MLflow
    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    # 7. Classification report to JSON
    cls_report = classification_report(y_test, y_pred_binary, output_dict=True)
    import json
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(cls_report, f, indent=4)
    mlflow.log_artifact(report_path)
    print(f"Classification report saved => {report_path}")

    # 8. threshold-check or fail if AUC < some level, do so here
    min_auc = config.get("min_auc", 0.7)
    if auc < min_auc:
        raise ValueError(f"AUC {auc:.3f} < minimum {min_auc} => failing pipeline for substandard model.")

    print(f"Evaluated model: {model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/evaluate.py config/eval_config.yaml")
        sys.exit(1)
    
    # open an MLflow run here so metrics go in a separate "evaluate" run
    mlflow.start_run(run_name="final_evaluation")
    main(sys.argv[1])
    mlflow.end_run()
