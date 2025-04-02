import yaml
import pandas as pd
import numpy as np
import os
import sys

def main(config_path):
    """Preprocess new loan data and merge with existing training data."""
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    drop_cols = config.get("drop_columns", [])
    remove_duplicates = config.get("remove_duplicates", True)
    newdata_folder = config.get("newdata_folder", "data/raw/new")
    base_train_path = config.get("base_train_path", "data/processed/train_loan.csv")

    # 1. Gather any new data files from the specified folder
    new_files = []
    if os.path.isdir(newdata_folder):
        new_files = [os.path.join(newdata_folder, f) for f in os.listdir(newdata_folder) if f.endswith(".csv")]
    if not new_files:
        print(f"No new CSV files found in {newdata_folder}.")
        # If no new data, ensure we have an existing base train file to use
        if not os.path.isfile(base_train_path):
            raise FileNotFoundError("No new data provided and no existing train_loan.csv found. Cannot preprocess data.")
        # Use the existing train_loan.csv as the latest processed data
        df_train = pd.read_csv(base_train_path)
        # Drop any columns not needed (e.g., prediction columns from prior runs)
        df_train.drop(columns=["prediction"], inplace=True, errors="ignore")
        output_path = os.path.join("data", "processed", "loan_clean_latest.csv")
        df_train.to_csv(output_path, index=False)
        print(f"No new data. Copied existing training data to {output_path}.")
        return

    # 2. Load and concatenate all new data CSV files
    df_new_list = [pd.read_csv(file) for file in new_files]
    new_data_df = pd.concat(df_new_list, ignore_index=True)
    print(f"Loaded {new_data_df.shape[0]} new rows from {len(new_files)} file(s) in {newdata_folder}.")

    # 3. Optionally merge with existing training data (for reference or removing overlaps)
    old_train_df = None
    if os.path.isfile(base_train_path):
        old_train_df = pd.read_csv(base_train_path)
        print(f"Existing training data loaded from {base_train_path} ({old_train_df.shape[0]} rows).")
    else:
        print("No existing training data found. New data will serve as the training set.")

    # 4. Remove duplicates in new data (and between new and old data if needed)
    if remove_duplicates:
        before_new = len(new_data_df)
        new_data_df.drop_duplicates(inplace=True)
        after_new = len(new_data_df)
        if after_new < before_new:
            print(f"Removed {before_new - after_new} duplicate rows in new data; {after_new} new rows remain.")
        # Also remove any duplicates that appear in both old and new (by full row match)
        if old_train_df is not None:
            combined = pd.concat([old_train_df, new_data_df], ignore_index=True)
            combined.drop_duplicates(inplace=True)
            # Determine how many new_data rows were dropped by duplicates with old data
            new_data_combined = combined.iloc[old_train_df.shape[0]:] if old_train_df.shape[0] > 0 else combined
            if len(new_data_combined) < len(new_data_df):
                removed = len(new_data_df) - len(new_data_combined)
                print(f"Removed {removed} new rows that duplicated existing training data.")
            new_data_df = new_data_combined if old_train_df.shape[0] == 0 else new_data_combined

    # 5. Filter new data by loanStatus (use same allowed statuses as in initial training)
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
    new_data_df.dropna(subset=["loanStatus"], inplace=True)
    new_data_df = new_data_df[new_data_df["loanStatus"].isin(allowed_statuses)]
    print(f"Filtered new data to {new_data_df.shape[0]} rows with allowed loanStatus values.")

    # 6. Create risk_indicator for new data (1 if positive outcome, else 0)
    new_data_df["risk_indicator"] = np.where(new_data_df["loanStatus"].isin(positive_statuses), 1, 0)

    # 7. Drop unwanted columns from new data
    new_data_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    new_data_df.drop(columns=["prediction"], inplace=True, errors="ignore")  # drop any prediction column if present

    # 8. Apply same encoding and feature engineering to new data as done for training data
    # Encode payFrequency
    if "payFrequency" in new_data_df.columns:
        if new_data_df["payFrequency"].mode().size > 0:
            mode_val = new_data_df["payFrequency"].mode()[0]
        else:
            mode_val = "W"
        new_data_df["payFrequency"] = new_data_df["payFrequency"].fillna(mode_val)
        payfreq_map = {"W": 0, "B": 1, "S": 2, "M": 3, "I": 4}
        new_data_df["payFrequency"] = new_data_df["payFrequency"].map(payfreq_map)
    # Encode leadType with fixed mapping
    if "leadType" in new_data_df.columns:
        new_data_df["leadType"] = new_data_df["leadType"].fillna("unknown")
        leadtype_map = {
            "bvMandatory": 0, "lead": 1, "california": 2, "organic": 3, "rc_returning": 4,
            "prescreen": 5, "express": 6, "repeat": 7, "instant-offer": 8, "unknown": 9
        }
        new_data_df["leadType"] = new_data_df["leadType"].apply(lambda x: x if x in leadtype_map else "unknown")
        new_data_df["leadType"] = new_data_df["leadType"].map(leadtype_map)
    # Map state to region_code
    if "state" in new_data_df.columns:
        new_data_df["state"] = new_data_df["state"].str.upper()
        region_map = {
            # (same mapping as in split_test.py)
            'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast', 'RI': 'Northeast',
            'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
            'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
            'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
            'ND': 'Midwest', 'SD': 'Midwest',
            'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South', 'SC': 'South',
            'VA': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South', 'MS': 'South', 'TN': 'South',
            'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South', 'DC': 'South',
            'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West',
            'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West', 'HI': 'West', 'OR': 'West', 'WA': 'West'
        }
        region_code_map = {'Northeast': 1, 'Midwest': 2, 'South': 3, 'West': 4}
        new_data_df["region_code"] = new_data_df["state"].map(lambda x: region_code_map.get(region_map.get(x), 0))
        new_data_df.drop(columns=["state"], inplace=True)
    # Compute interest_pct feature
    if "loanAmount" in new_data_df.columns and "originallyScheduledPaymentAmount" in new_data_df.columns:
        new_data_df["interest_pct"] = new_data_df["originallyScheduledPaymentAmount"].fillna(new_data_df["loanAmount"]) / new_data_df["loanAmount"]
        new_data_df["interest_pct"] = new_data_df["interest_pct"].replace([np.inf, -np.inf], np.nan) - 1
        new_data_df["interest_pct"] = new_data_df["interest_pct"].fillna(0)
    # Drop loanStatus column from new data after creating risk_indicator
    new_data_df.drop(columns=["loanStatus"], inplace=True, errors="ignore")

    # 9. Merge new processed data with existing training data
    if old_train_df is not None:
        # Ensure old training data has no 'prediction' or other unused columns
        old_train_df.drop(columns=["prediction"], inplace=True, errors="ignore")
        # Concatenate old and new data (they should have identical columns now)
        df_combined = pd.concat([old_train_df, new_data_df], ignore_index=True)
        # Remove any duplicates after merging (if the same loan appears in old and new)
        if remove_duplicates:
            before_merge = len(df_combined)
            df_combined.drop_duplicates(inplace=True)
            after_merge = len(df_combined)
            if after_merge < before_merge:
                print(f"Removed {before_merge - after_merge} duplicate rows after merging new data with existing training data.")
    else:
        df_combined = new_data_df

    # 10. Save the updated processed dataset for model training
    os.makedirs("data/processed", exist_ok=True)
    output_path = os.path.join("data", "processed", "loan_clean_latest.csv")
    df_combined.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Saved merged data to {output_path} (total {df_combined.shape[0]} rows).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py config/preprocess_config.yaml")
        sys.exit(1)
    main(sys.argv[1])
