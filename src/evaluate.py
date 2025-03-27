# src/evaluate.py

import os
import sys
import yaml
import glob
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import warnings

from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or your MLflow server
mlflow.set_experiment("LoanRisk")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

def compute_metric(metric_name, y_true, y_prob, y_bin):
    if metric_name == "auc":
        return roc_auc_score(y_true, y_prob)
    elif metric_name == "accuracy":
        return accuracy_score(y_true, y_bin)
    elif metric_name == "precision":
        return precision_score(y_true, y_bin)
    elif metric_name == "recall":
        return recall_score(y_true, y_bin)
    elif metric_name == "f1":
        return f1_score(y_true, y_bin)
    else:
        raise ValueError(f"Unsupported metric => {metric_name}")

def delete_alias_if_exists(client, model_name, alias_name):
    """Remove alias_name from whichever version currently has it."""
    all_mvs = client.search_model_versions(f"name='{model_name}'")
    for mv in all_mvs:
        if hasattr(mv, "aliases") and mv.aliases:
            if alias_name in mv.aliases:
                client.delete_model_version_alias(model_name, mv.version, alias_name)
                print(f"Removed alias '{alias_name}' from version {mv.version}")
                return

def load_new_model_artifact(models_dir):
    pattern_txt = os.path.join(models_dir, "*.txt")
    pattern_pkl = os.path.join(models_dir, "*.pkl")
    candidates = glob.glob(pattern_txt) + glob.glob(pattern_pkl)
    if not candidates:
        raise FileNotFoundError(f"No local model artifacts in => {models_dir}")
    newest = max(candidates, key=os.path.getmtime)
    print(f"Newest local artifact => {newest}")
    if newest.endswith(".txt"):
        booster = lgb.Booster(model_file=newest)
        return booster, newest
    else:
        import pickle
        with open(newest, "rb") as f:
            clf = pickle.load(f)
        return clf, newest

def load_alias(model_name, alias_name="production"):
    client = MlflowClient()
    alias_uri = f"models:/{model_name}@{alias_name}"
    try:
        champion_model = mlflow.pyfunc.load_model(alias_uri)
    except Exception:
        return None, None
    # find which version has that alias
    all_mvs = client.search_model_versions(f"name='{model_name}'")
    champion_version = None
    for mv in all_mvs:
        if alias_name in getattr(mv, "aliases", []):
            champion_version = mv.version
            break
    return champion_model, champion_version

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    test_data_path = config.get("test_data_path","data/test/test_loan.csv")
    if not os.path.isfile(test_data_path):
        raise FileNotFoundError(f"No test data => {test_data_path}")

    df = pd.read_csv(test_data_path)
    target_col = config.get("target_col","risk_indicator")
    if target_col not in df.columns:
        raise ValueError("target col not found in test dataset")

    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    # load new local model
    models_dir = config.get("models_dir","models")
    new_model, new_path = load_new_model_artifact(models_dir)

    # predictions
    if isinstance(new_model, lgb.Booster):
        y_prob_new = new_model.predict(X_test)
    else:
        if hasattr(new_model,"predict_proba"):
            y_prob_new = new_model.predict_proba(X_test)[:,1]
        else:
            y_prob_new = new_model.predict(X_test)

    threshold = config.get("threshold_for_class", 0.5)
    y_bin_new = (y_prob_new > threshold).astype(int)

    # basic metrics
    auc_new = roc_auc_score(y_test, y_prob_new)
    acc_new = accuracy_score(y_test, y_bin_new)
    pre_new = precision_score(y_test, y_bin_new)
    rec_new = recall_score(y_test, y_bin_new)
    f1_new = f1_score(y_test, y_bin_new)
    print(f"New model test => AUC={auc_new:.4f}, ACC={acc_new:.4f}, PREC={pre_new:.4f}, REC={rec_new:.4f}, F1={f1_new:.4f}")

    mlflow.log_metric("test_auc", auc_new)
    mlflow.log_metric("test_accuracy", acc_new)
    mlflow.log_metric("test_precision", pre_new)
    mlflow.log_metric("test_recall", rec_new)
    mlflow.log_metric("test_f1", f1_new)

    # classification report
    from sklearn.metrics import classification_report
    cls_rep = classification_report(y_test, y_bin_new, output_dict=True)

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports","classification_report.json")
    with open(report_path,"w") as rf:
        json.dump(cls_rep, rf, indent=4)
    mlflow.log_artifact(report_path)

    # 1) min metric check
    min_metric_name = config.get("min_metric_name","auc")
    min_metric_val  = config.get("min_metric_value",0.7)
    new_val_min = compute_metric(min_metric_name, y_test, y_prob_new, y_bin_new)
    print(f"[Threshold Check] {min_metric_name}={new_val_min:.4f} => needs >= {min_metric_val:.4f}")
    if new_val_min < min_metric_val:
        raise ValueError(f"New model's {min_metric_name}={new_val_min:.3f} < {min_metric_val:.3f}, failing pipeline => no promotion")

    # 2) champion-challenger using compare_metric
    compare_metric = config.get("compare_metric","auc")
    new_val_cmp = compute_metric(compare_metric, y_test, y_prob_new, y_bin_new)
    print(f"[Champion-challenger] new model {compare_metric} => {new_val_cmp:.4f}")

    client = MlflowClient()
    champion_model, champion_version = load_alias("LoanRiskModel", "production")
    if champion_model is None:
        print("No production alias => new model is champion by default.")
        do_promote = True
        champion_version = None
    else:
        # champion predictions
        champ_preds = champion_model.predict(X_test)
        if champ_preds.ndim==2 and champ_preds.shape[1]==2:
            champ_prob = champ_preds[:,1]
        else:
            champ_prob = champ_preds
        champ_bin = (champ_prob > threshold).astype(int)
        champ_val_cmp = compute_metric(compare_metric, y_test, champ_prob, champ_bin)
        print(f"Champion => alias=production, ver={champion_version}, {compare_metric}={champ_val_cmp:.4f}")

        do_promote = (new_val_cmp > champ_val_cmp)

    if do_promote:
        # read new version from model_version.txt
        ver_file = "model_version.txt"
        if not os.path.isfile(ver_file):
            print("No model_version.txt => skip promotion.")
        else:
            with open(ver_file) as fv:
                new_ver = fv.read().strip()
            new_ver = int(new_ver)

            # set alias=production on the new version
            client.set_registered_model_alias(
                name="LoanRiskModel",
                alias="production",
                version=new_ver
            )
            print(f"Promoted => production alias => version {new_ver}. Old v{champion_version} removed alias.")
    else:
        raise ValueError("Champion is better => pipeline fail => no promotion")

    print(f"Done => new model artifact: {new_path}")

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python evaluate.py config/eval_config.yaml")
        sys.exit(1)
    mlflow.start_run(run_name="final_evaluation")
    main(sys.argv[1])
    mlflow.end_run()
