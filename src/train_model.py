import os
import sys
import yaml
import pickle
import datetime
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier

def train_default(X_train, y_train, X_val, y_val, config, random_state, timestamp):
    """Train a LightGBM model with default parameters from config."""
    # Prepare LightGBM Dataset objects
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    # Base parameters for LightGBM model
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": config.get("learning_rate", 0.05),
        "num_leaves": config.get("num_leaves", 31),
        "verbose": config.get("verbose", -1),
        "seed": random_state
    }
    # Log parameters to MLflow
    mlflow.log_params(params)
    mlflow.log_param("test_size", config.get("test_size", 0.3))
    mlflow.log_param("random_state", random_state)
    # Train with early stopping
    early_stopping_cb = lgb.early_stopping(stopping_rounds=config.get("early_stopping_rounds", 50))
    log_eval_cb = lgb.log_evaluation(period=100)
    model = lgb.train(
        params, train_data,
        num_boost_round=config.get("num_boost_round", 1000),
        valid_sets=[val_data],
        callbacks=[early_stopping_cb, log_eval_cb]
    )
    # Predict probabilities on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return model, y_pred

def train_grid_search(X_train, y_train, X_val, y_val, config, random_state, timestamp):
    """Train a LightGBM model using grid search for hyperparameter tuning."""
    # Parameter grid for GridSearchCV
    param_grid = config.get("param_grid", {
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63]
    })
    base_estimator = LGBMClassifier(
        objective="binary",
        n_estimators=config.get("num_boost_round", 1000),
        random_state=random_state
    )
    grid = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring=config.get("grid_scoring", "roc_auc"),
        cv=config.get("grid_cv", 3),
        verbose=config.get("grid_verbose", 1),
        n_jobs=config.get("grid_n_jobs", -1)
    )
    grid.fit(X_train, y_train)
    # Log best parameters to MLflow
    mlflow.log_params(grid.best_params_)
    best_model = grid.best_estimator_
    # Predict probabilities on validation set
    y_pred = best_model.predict_proba(X_val)[:, 1]
    # Save the best model to a file
    model_dir = config.get("model_dir", "output/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{config.get('grid_model_prefix', 'lightgbm_best_model')}_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    # Log the model to MLflow (sklearn flavor) with signature
    mlflow.sklearn.log_model(
        best_model, artifact_path="model",
        registered_model_name="LoanRiskModel",
        input_example=X_val[:5],
        signature=mlflow.models.infer_signature(X_val, y_val)
    )
    # Save grid search results as artifact
    results_df = pd.DataFrame(grid.cv_results_)
    results_dir = config.get("grid_search_results_dir", "output/grid_search_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"grid_search_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path, artifact_path="grid_search_results")
    return best_model, y_pred, model_path

def main(config_path):
    """Train the LightGBM model and log results and model artifacts using MLflow."""
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Set up MLflow tracking URI and experiment 
    tracking_uri = config.get("tracking_uri", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("experiment_name", "LoanRisk"))
    # Load the processed dataset
    dataset_path = config.get("dataset_path", "data/processed/loan_clean_latest.csv")
    print(f"Reading training dataset from {dataset_path}")
    data = pd.read_csv(dataset_path)
    target_col = config.get("target_col", "risk_indicator")
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    # Split data into training and validation sets
    test_size = config.get("test_size", 0.3)
    random_state = config.get("random_state", 42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
    # Start MLflow run and train model
    mlflow.start_run()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if config.get("grid_search", False):
        print("Performing grid search hyperparameter tuning...")
        model, y_pred, model_path = train_grid_search(X_train, y_train, X_val, y_val, config, random_state, timestamp)
    else:
        print("Training model with default hyperparameters...")
        model, y_pred = train_default(X_train, y_train, X_val, y_val, config, random_state, timestamp)
        # Save model to file
        model_dir = config.get("model_dir", "output/models")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{config.get('model_prefix', 'lightgbm_model')}_{timestamp}.txt")
        model.save_model(model_file)
        # Log the LightGBM model to MLflow (LightGBM flavor)
        mlflow.lightgbm.log_model(
            model, artifact_path="model",
            registered_model_name="LoanRiskModel",
            input_example=X_val[:5],
            signature=mlflow.models.infer_signature(X_val, y_val)
        )
        model_path = model_file
    # Evaluate model on validation set
    auc = roc_auc_score(y_val, y_pred)
    threshold = config.get("threshold", 0.5)
    y_pred_labels = (y_pred > threshold).astype(int)
    accuracy = accuracy_score(y_val, y_pred_labels)
    precision = precision_score(y_val, y_pred_labels)
    recall = recall_score(y_val, y_pred_labels)
    # Log evaluation metrics to MLflow
    mlflow.log_metric("val_auc", auc)
    mlflow.log_metric("val_accuracy", accuracy)
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)
    mlflow.end_run()
    # Update model registry alias
    client = MlflowClient()
    versions = client.search_model_versions(f"name='LoanRiskModel'")
    if versions:
        new_version = max(int(mv.version) for mv in versions)
        alias_name = "candidate"  # mark the newest model as "candidate"
        client.set_registered_model_alias(name="LoanRiskModel", alias=alias_name, version=new_version)
        with open("model_version.txt", "w") as f:
            f.write(str(new_version))
        print(f"Registered model version {new_version} as alias '{alias_name}'.")
    # Print out final metrics and model path
    print("Model training complete.")
    print(f"Validation AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Model artifact saved to: {model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py config/train_config.yaml")
        sys.exit(1)
    main(sys.argv[1])
