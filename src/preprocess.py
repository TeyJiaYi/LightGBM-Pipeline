import datetime
import yaml
import pandas as pd
import os
import sys
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder

# src/preprocess.py

def main(config_path):
    
    # load config YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    drop_cols = config.get("drop_columns",[])
    df.drop(columns=['prediction'], inplace=True, errors='ignore')
    remove_duplicates = config.get("remove_duplicates", True)
    newdata_folder = config.get("newdata_folder", "data/raw/new")

    # stable training set from split_test
    base_train_path = config.get("base_train_path", "train_loan.csv")


    # 2. Gather CSVs from newdata_folder
    csv_files = glob.glob(os.path.join(newdata_folder, "*.csv"))
    if not csv_files:
        print(f"No new CSV data found in: {newdata_folder}")

        # If no new data, rely on the stable training file
        if not os.path.isfile(base_train_path):
            # If there's also no train_loan.csv, we must fail:
            raise FileNotFoundError(
                "No new data AND no existing train_loan.csv. Cannot produce loan_clean_latest.csv."
            )

        # Otherwise, proceed with only train_loan.csv for the final output
        print(f"Using existing {base_train_path} for 'loan_clean_latest.csv' since no new data.")
        df = pd.read_csv(base_train_path)
        os.makedirs("data/processed", exist_ok=True)
        processed_data_path = os.path.join("data", "processed", f"loan_clean_latest.csv")
        df.to_csv(processed_data_path, index=False)
        return

    else:
        # 3. We do have new data, read & unify with train_loan.csv if exists
        df_list = []
        for file in csv_files:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        new_data_df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {new_data_df.shape[0]} rows from {len(csv_files)} file(s) in {newdata_folder}.")

        if os.path.isfile(base_train_path):
            print(f"Merging existing base training data from: {base_train_path}")
            old_train_df = pd.read_csv(base_train_path)
            df = pd.concat([old_train_df, new_data_df], ignore_index=True)
            print(f"Combined old train + new data => total {df.shape[0]} rows.")
        else:
            # If no stable train_loan yet, treat new_data as entire training data
            df = new_data_df
            print("No base training file found; using only new data as training set.")

    # 5. Remove duplicates if specified
    if remove_duplicates:
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        print(f"Dropped {before - after} duplicates; final {after} rows.")

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
    positive_loanstatus_list = config.get("positive_loanstatus",default_positive_statuses)
    negative_loanstatus_list = config.get("negative_loanstatus", default_negative_statuses)

    positive_loanstatus = set(positive_loanstatus_list)
    negative_loanstatus = set(negative_loanstatus_list)

    df.dropna(subset=['loanStatus'], inplace=True)
    allowed_statuses = positive_loanstatus.union(negative_loanstatus)
    df = df[df['loanStatus'].isin(allowed_statuses)]

    # 4. Create the new risk indicator (binary label)
    df['risk_indicator'] = np.where(
    df['loanStatus'].isin(positive_loanstatus), 
    1,  # Positive outcome
    0   # Negative outcome
    )

    df.drop(columns='loanStatus', inplace=True)
    

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    
    # Encoding fot the payfrequency
    if "payFrequency" in df.columns:
        mode_value = df["payFrequency"].mode()[0]
        df["payFrequency"] = df["payFrequency"].fillna(mode_value)

        payfreq_map = {
            "W": 0,  # Weekly
            "B": 1,  # Bi-weekly
            "S": 2,  # Semi-monthly
            "M": 3,  # Monthly
            "I": 4   # Irregular
        }
        df["payFrequency"] = df["payFrequency"].map(payfreq_map)

    if "leadType" in df.columns:
        df["leadType"] = df["leadType"].fillna("unknown")
        encoder = LabelEncoder()
        df["leadType"] = encoder.fit_transform(df["leadType"])

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

    # 2) Define numeric codes
    region_code_map = {
        'Northeast': 1,
        'Midwest': 2,
        'South': 3,
        'West': 4
    }

    if "state" in df.columns:
        df["state"] = df["state"].str.upper()
        df["region_code"] = df["state"].map(lambda x: region_code_map.get(region_map.get(x), None))
        df.drop(columns=["state"], inplace=True)

    # save the file
    os.makedirs("data/processed", exist_ok=True)
    processed_data_path = os.path.join("data", "processed", f"loan_clean_latest.csv")
    df.to_csv(processed_data_path, index=False)

    print(f"Preprocessing complete. Overwrote => {processed_data_path} with merged data, total rows={len(df)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/preprocess.py config/preprocess_config.yaml")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)
