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

    data_path = config.get("data_path", "data/raw/loan.csv")
    drop_cols = config.get("drop_columns",[])
    remove_duplicates = config.get("remove_duplicates", True)
    newdata_folder = config.get("newdata_folder", "data/raw/new")


    df = pd.read_csv(data_path)

    csv_files = glob.glob(os.path.join(newdata_folder, "*.csv"))
    if not csv_files:
        print(f"No new CSV data files found in the folder: {newdata_folder}")

    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)

    if remove_duplicates:
        df.drop_duplicates(inplace=True)

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

    
    # Encoding fot he payfrequency
    if "payFrequency" in df.columns:
        mode_value = df["payFrequency"].mode()[0]
        df["payFrequency"].fillna(mode_value, inplace=True)

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

    df['state'] = df['state'].str.upper()
    df['region_code'] = df['state'].map(lambda x: region_code_map.get(region_map.get(x), None))
    df.drop(columns=['state'], inplace=True)


    os.makedirs("data/processed", exist_ok=True)
    processed_data_path = os.path.join("data","processed","loan_clean.csv")
    df.to_csv(processed_data_path, index=False)

    print(f"Preprocessing complete. Processed file saved to: {processed_data_path}")

if __name__ == "__main__":
    # usage: python src/preprocess.py config/preprocess_config.yaml
    config_file = sys.argv[1]
    main(config_file)
