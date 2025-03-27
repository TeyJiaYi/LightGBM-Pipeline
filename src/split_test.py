import datetime
import yaml
import pandas as pd
import os
import sys
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def main(config_path):
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Read all CSVs from original_data_folder
    original_data_folder = config.get("original_data_folder", "data/raw/original")
    csv_files = glob.glob(os.path.join(original_data_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {original_data_folder}. Exiting.")
        sys.exit(0)

    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {df.shape[0]} rows from {len(csv_files)} CSV file(s) in {original_data_folder}")

    # 3. Remove duplicates if specified
    remove_duplicates = config.get("remove_duplicates", True)
    if remove_duplicates:
        df.drop_duplicates(inplace=True)

    # 4. Filter out rows based on loanStatus
    default_positive_statuses = ["Paid Off Loan"]
    default_negative_statuses = [
        "External Collection", 
        "Internal Collection",
        "Returned Item",
        "Settlement Paid Off",
        "Settled Bankruptcy",
        "Charged Off",
        "Charged Off Paid Off"
    ]
    positive_loanstatus_list = config.get("positive_loanstatus", default_positive_statuses)
    negative_loanstatus_list = config.get("negative_loanstatus", default_negative_statuses)
    positive_set = set(positive_loanstatus_list)
    negative_set = set(negative_loanstatus_list)

    df.dropna(subset=["loanStatus"], inplace=True)
    allowed_statuses = positive_set.union(negative_set)
    df = df[df["loanStatus"].isin(allowed_statuses)]

    # 5. Create risk_indicator
    df["risk_indicator"] = np.where(
        df["loanStatus"].isin(positive_set),
        1,
        0
    )
    df.drop(columns="loanStatus", inplace=True, errors="ignore")

    # 6. Drop user-specified columns
    drop_cols = config.get("drop_columns", [])
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 7. Encode payFrequency
    if "payFrequency" in df.columns:
        mode_value = df["payFrequency"].mode()[0]
        df["payFrequency"] = df["payFrequency"].fillna(mode_value)
        payfreq_map = {
            "W": 0,
            "B": 1,
            "S": 2,
            "M": 3,
            "I": 4
        }
        df["payFrequency"] = df["payFrequency"].map(payfreq_map)

    # 8. Encode leadType
    if "leadType" in df.columns:
        df["leadType"] = df["leadType"].fillna("unknown")
        encoder = LabelEncoder()
        df["leadType"] = encoder.fit_transform(df["leadType"])

    # 9. Map state -> region_code
    region_map = {
        # Northeast
        'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast', 'RI': 'Northeast',
        'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
        # Midwest
        'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
        'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
        'ND': 'Midwest', 'SD': 'Midwest',
        # South
        'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
        'SC': 'South', 'VA': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South',
        'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South',
        'TX': 'South', 'DC': 'South',
        # West
        'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
        'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
        'HI': 'West', 'OR': 'West', 'WA': 'West'
    }
    region_code_map = {'Northeast': 1, 'Midwest': 2, 'South': 3, 'West': 4}

    if "state" in df.columns:
        df["state"] = df["state"].str.upper()
        df["region_code"] = df["state"].map(lambda x: region_code_map.get(region_map.get(x), None))
        df.drop(columns=["state"], inplace=True)

    # 10. Split into train/test
    test_ratio = config.get("test_ratio", 0.2)
    random_state = config.get("random_state", 42)

    y = df["risk_indicator"]
    X = df.drop(columns=["risk_indicator"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state
    )

    # 11. (Optional) sampling on the *train set* only

    balance_mode = config.get("balance_mode", "none")  # "none", "undersample", "oversample", or "smote"

    df_train = X_train.copy()
    df_train["risk_indicator"] = y_train

    if balance_mode == "undersample":
        class0 = df_train[df_train["risk_indicator"] == 0]
        class1 = df_train[df_train["risk_indicator"] == 1]
        min_class_size = min(len(class0), len(class1))
        class0_under = class0.sample(n=min_class_size, random_state=random_state)
        class1_under = class1.sample(n=min_class_size, random_state=random_state)
        df_train = pd.concat([class0_under, class1_under], ignore_index=True)
        print(f"Undersampled train set to have equal positives/negatives => total {len(df_train)} rows")
    elif balance_mode == "oversample":
        class0 = df_train[df_train["risk_indicator"] == 0]
        class1 = df_train[df_train["risk_indicator"] == 1]
        max_class_size = max(len(class0), len(class1))
        if len(class0) < max_class_size:
            class0 = class0.sample(n=max_class_size, replace=True, random_state=random_state)
        if len(class1) < max_class_size:
            class1 = class1.sample(n=max_class_size, replace=True, random_state=random_state)
        df_train = pd.concat([class0, class1], ignore_index=True)
        print(f"Oversampled train set => total {len(df_train)} rows")
    elif balance_mode == "smote":
        smote = SMOTE(random_state=random_state)
        features = df_train.drop(columns=["risk_indicator"])
        labels = df_train["risk_indicator"]
        X_sm, y_sm = smote.fit_resample(features, labels)
        df_train = X_sm.copy()
        df_train["risk_indicator"] = y_sm
        print(f"Applied SMOTE; new train set size: {len(df_train)}")
    else:
        print("No sampling performed on the train set.")

    df_test = X_test.copy()
    df_test["risk_indicator"] = y_test

    # 12. Save outputs
    os.makedirs("data/processed", exist_ok=True)
    train_path = os.path.join("data", "processed", "train_loan.csv")
    df_train.to_csv(train_path, index=False)
    print(f"Train dataset => {train_path}, rows={len(df_train)}")

    os.makedirs("data/test", exist_ok=True)
    test_path = os.path.join("data", "test", "test_loan.csv")
    df_test.to_csv(test_path, index=False)
    print(f"Test dataset => {test_path}, rows={len(df_test)}")

    print("split_test.py complete! This is run once or rarely, so your final test set remains stable.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/split_test.py config/split_test_config.yaml")
        sys.exit(1)
    main(sys.argv[1])
