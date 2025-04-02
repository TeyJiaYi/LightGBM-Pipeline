import datetime
import yaml
import pandas as pd
import numpy as np
import os
import sys
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def main(config_path):
    """Split original loan data into training and test sets with preprocessing."""
    # 1. Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Read all CSV files from the original data folder
    original_data_folder = config.get("original_data_folder", "data/raw/original")
    csv_files = glob.glob(os.path.join(original_data_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {original_data_folder}. Exiting.")
        sys.exit(0)
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {df.shape[0]} rows from {len(csv_files)} file(s) in {original_data_folder}")

    # 3. Remove duplicate rows if specified
    if config.get("remove_duplicates", True):
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        if after < before:
            print(f"Removed {before - after} duplicate rows; {after} unique rows remain.")

    # 4. Filter rows by loanStatus (keep only specified positive/negative statuses)
    default_positive = ["Paid Off Loan", "Pending Paid Off"]
    default_negative = [
        "External Collection", "Internal Collection",
        "Settlement Paid Off", "Settled Bankruptcy",
        "Charged Off", "Charged Off Paid Off",
        "Settlement Pending Paid Off"
    ]
    positive_statuses = set(config.get("positive_loanstatus", default_positive))
    negative_statuses = set(config.get("negative_loanstatus", default_negative))
    allowed_statuses = positive_statuses.union(negative_statuses)
    df.dropna(subset=["loanStatus"], inplace=True)
    df = df[df["loanStatus"].isin(allowed_statuses)]
    print(f"Filtered to {df.shape[0]} rows with allowed loanStatus values.")

    # 5. Create binary target column `risk_indicator` (1 for positive outcome, 0 for negative)
    df["risk_indicator"] = np.where(df["loanStatus"].isin(positive_statuses), 1, 0)

    # 6. Drop user-specified columns that won't be used as features
    drop_cols = config.get("drop_columns", [])
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 7. Encode payFrequency (fill missing with mode, map categories to numeric order)
    if "payFrequency" in df.columns:
        if df["payFrequency"].mode().size > 0:
            mode_val = df["payFrequency"].mode()[0]
        else:
            mode_val = "W"
        df["payFrequency"] = df["payFrequency"].fillna(mode_val)
        payfreq_map = {"W": 0, "B": 1, "S": 2, "M": 3, "I": 4}
        df["payFrequency"] = df["payFrequency"].map(payfreq_map)

    # 8. Encode leadType (fill missing with "unknown", then map each category to a fixed integer)
    if "leadType" in df.columns:
        df["leadType"] = df["leadType"].fillna("unknown")
        # Define a fixed mapping for leadType categories to ensure consistency
        leadtype_map = {
            "bvMandatory": 0, "lead": 1, "california": 2, "organic": 3, "rc_returning": 4,
            "prescreen": 5, "express": 6, "repeat": 7, "instant-offer": 8, "unknown": 9
        }
        df["leadType"] = df["leadType"].apply(lambda x: x if x in leadtype_map else "unknown")
        df["leadType"] = df["leadType"].map(leadtype_map)

    # 9. Map state to region_code (group states into regions and encode as numeric)
    if "state" in df.columns:
        df["state"] = df["state"].str.upper()
        region_map = {
            # Northeast
            'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
            'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
            # Midwest
            'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
            'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
            'ND': 'Midwest', 'SD': 'Midwest',
            # South
            'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South', 'SC': 'South',
            'VA': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South', 'MS': 'South', 'TN': 'South',
            'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South', 'DC': 'South',
            # West
            'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West',
            'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West', 'HI': 'West', 'OR': 'West', 'WA': 'West'
        }
        region_code_map = {'Northeast': 1, 'Midwest': 2, 'South': 3, 'West': 4}
        df["region_code"] = df["state"].map(lambda x: region_code_map.get(region_map.get(x), 0))
        df.drop(columns=["state"], inplace=True)

    # 10. Create derived feature: interest_pct (percentage interest = (total scheduled payment / loanAmount) - 1)
    if "loanAmount" in df.columns and "originallyScheduledPaymentAmount" in df.columns:
        df["interest_pct"] = df["originallyScheduledPaymentAmount"].fillna(df["loanAmount"]) / df["loanAmount"]
        df["interest_pct"] = df["interest_pct"].replace([np.inf, -np.inf], np.nan) - 1
        df["interest_pct"] = df["interest_pct"].fillna(0)


    # 11. Drop the original loanStatus column (not needed after creating risk_indicator)
    df.drop(columns=["loanStatus"], inplace=True, errors="ignore")

    # 12. Split data into train and test sets with stratification on risk_indicator
    test_ratio = config.get("test_ratio", 0.2)
    random_state = config.get("random_state", 42)
    X = df.drop(columns=["risk_indicator"])
    y = df["risk_indicator"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=random_state
    )
    df_train = X_train.copy()
    df_train["risk_indicator"] = y_train
    df_test = X_test.copy()
    df_test["risk_indicator"] = y_test
    print(f"Split data: {df_train.shape[0]} rows for training, {df_test.shape[0]} rows for testing.")

    # 13. Perform optional class balancing on the training set only
    balance_mode = config.get("balance_mode", "none").lower()  # "none", "undersample", "oversample", or "smote"
    if balance_mode == "undersample":
        class0 = df_train[df_train["risk_indicator"] == 0]
        class1 = df_train[df_train["risk_indicator"] == 1]
        n = min(len(class0), len(class1))
        df_train = pd.concat([class0.sample(n=n, random_state=random_state),
                               class1.sample(n=n, random_state=random_state)], ignore_index=True)
        print(f"Performed undersampling: Train set now {df_train.shape[0]} rows (balanced).")
    elif balance_mode == "oversample":
        class0 = df_train[df_train["risk_indicator"] == 0]
        class1 = df_train[df_train["risk_indicator"] == 1]
        n = max(len(class0), len(class1))
        if len(class0) < n:
            class0 = class0.sample(n=n, replace=True, random_state=random_state)
        if len(class1) < n:
            class1 = class1.sample(n=n, replace=True, random_state=random_state)
        df_train = pd.concat([class0, class1], ignore_index=True)
        print(f"Performed oversampling: Train set now {df_train.shape[0]} rows (balanced).")
    elif balance_mode == "smote":
        smote = SMOTE(random_state=random_state)
        features = df_train.drop(columns=["risk_indicator"]).fillna(0)
        labels = df_train["risk_indicator"]
        X_sm, y_sm = smote.fit_resample(features, labels)
        df_train = X_sm.copy()
        df_train["risk_indicator"] = y_sm
        print(f"Applied SMOTE: Train set now {df_train.shape[0]} rows (balanced).")
    else:
        print("No class balancing applied to the training set.")

    # 14. Save the processed training and test sets to CSV files
    os.makedirs("data/processed", exist_ok=True)
    train_path = os.path.join("data", "processed", "train_loan.csv")
    df_train.to_csv(train_path, index=False)
    print(f"Training dataset saved to {train_path} ({df_train.shape[0]} rows).")
    os.makedirs("data/test", exist_ok=True)
    test_path = os.path.join("data", "test", "test_loan.csv")
    df_test.to_csv(test_path, index=False)
    print(f"Test dataset saved to {test_path} ({df_test.shape[0]} rows).")
    print("split_test.py complete: stable test set created and saved.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_test.py config/split_test_config.yaml")
        sys.exit(1)
    main(sys.argv[1])
