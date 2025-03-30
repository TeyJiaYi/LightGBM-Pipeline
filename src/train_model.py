
import os
import sys
import yaml
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import datetime
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import pickle

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LoanRisk")  


def train_default(X_train, y_train, X_val, y_val, config, random_state, timestamp):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": config.get("learning_rate", 0.05),
        "num_leaves": config.get("num_leaves", 31),
        "verbose": config.get("verbose", -1),  # silent mode
        "seed": random_state,
    }

    # Log base parameters
    mlflow.log_params(params)
    mlflow.log_param("test_size", config.get("test_size", 0.3))
    mlflow.log_param("random_state", random_state)
    early_stopping_cb = lgb.early_stopping(stopping_rounds=config.get('early_stopping_rounds', 50))
    log_eval_cb = lgb.log_evaluation(period=100)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.get("num_boost_round", 1000),
        valid_sets=[val_data],
        callbacks=[early_stopping_cb, log_eval_cb]
    )
    
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return model, y_pred

def train_grid_search(X_train, y_train, X_val, y_val, config, random_state, timestamp):
    
    param_grid = config.get("param_grid", 
                            {"learning_rate": [0.01, 0.05, 0.1], 
                             "num_leaves": [15, 31, 63]})
    
    lgbm_clf = LGBMClassifier(
        objective="binary",
        random_state=random_state,
        n_estimators=config.get("num_boost_round", 1000)
    )

    cv = config.get("grid_cv", 3)
    verbose = config.get("grid_verbose", 1)
    n_jobs = config.get("grid_n_jobs", -1)
    scoring = config.get("grid_scoring", "roc_auc")
    
    grid = GridSearchCV(
        estimator=lgbm_clf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs
    )
    
    grid.fit(X_train, y_train)
    
    mlflow.log_params(grid.best_params_)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict_proba(X_val)[:, 1]
    
    # Save the best model artifact using the filename from YAML and the timestamp
    model_dir = config.get("model_dir", "output/models")
    os.makedirs(model_dir, exist_ok=True)
    grid_model_prefix = config.get("grid_model_prefix", "lightgbm_best_model")
    model_path = os.path.join(model_dir, f"{grid_model_prefix}_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    
    # Log the model via MLflow using the sklearn flavor
    
    mlflow.sklearn.log_model(best_model, 
                             "model", 
                             registered_model_name="LoanRiskModel",
                             input_example=X_val[:5],
                             signature=mlflow.models.infer_signature(X_val, y_val),)

    
    # Save grid search results to a CSV file and log as an artifact
    grid_results_df = pd.DataFrame(grid.cv_results_)
    grid_results_dir = config.get("grid_search_results_dir", "output/grid_search_results")
    os.makedirs(grid_results_dir, exist_ok=True)
    grid_results_path = os.path.join(grid_results_dir, f"grid_search_results_{timestamp}.csv")
    grid_results_df.to_csv(grid_results_path, index=False)
    mlflow.log_artifact(grid_results_path, artifact_path="grid_search_results")
    
    return best_model, y_pred, model_path

def main(config_path):
    # Load config YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_path = config.get("dataset_path", "data/processed/loan_clean_latest.csv")
    print(f"Reading merged dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path)

    target_col = config.get("target_col", "risk_indicator")
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]

    test_size = config.get("test_size", 0.3)
    random_state = config.get("random_state", 42)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create a timestamp for dynamic naming (format: YYYYMMDDHHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    mlflow.start_run()
    
    if config.get("grid_search", False):
        print("Performing grid search...")
        model, y_pred, model_path = train_grid_search(X_train, y_train, X_val, y_val, config, random_state, timestamp)
    else:
        print("Training with default parameters...")
        model, y_pred = train_default(X_train, y_train, X_val, y_val, config, random_state, timestamp)
        os.makedirs(config.get("model_dir", "output/models"), exist_ok=True)
        model_prefix = config.get("model_prefix", "lightgbm_model")
        model_path = os.path.join(config.get("model_dir", "output/models"), f"{model_prefix}_{timestamp}.txt")
        model.save_model(model_path)
        mlflow.lightgbm.log_model(model, "model", 
                                  registered_model_name="LoanRiskModel", 
                                  input_example=X_val[:5],
                                  signature=mlflow.models.infer_signature(X_val, y_val),)

    auc = roc_auc_score(y_val, y_pred)
    threshold = config.get("threshold", 0.5)
    y_pred_binary = (y_pred > threshold).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)

    mlflow.log_metric("val_auc", auc)
    mlflow.log_metric("val_accuracy", accuracy)
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)

    mlflow.end_run()


    client = MlflowClient()

    # new (alias-based) lines
    all_versions = client.search_model_versions(f"name='LoanRiskModel'")
    if not all_versions:
        print("No model versions found. Cannot set alias.")
    else:
        new_version = max(int(mv.version) for mv in all_versions)
        print(f"Newest version => {new_version}")
        alias_name = "candidate"   # or "latest"
        client.set_registered_model_alias(
            name="LoanRiskModel",
            alias=alias_name,
            version=new_version
        )
        with open("model_version.txt","w") as f:
            f.write(str(new_version))


    print("Model training complete.")
    print(f"Validation AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/train_model.py config/train_config.yaml")
        sys.exit(1)
    main(sys.argv[1])
