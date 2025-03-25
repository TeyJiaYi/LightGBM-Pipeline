import yaml
import pandas as pd
import os
import sys
import glob

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

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    os.makedirs("data/processed", exist_ok=True)
    processed_data_path = os.path.join("data","processed","loan_clean.csv")
    df.to_csv(processed_data_path, index=False)

    print(f"Preprocessing complete. Processed file saved to: {processed_data_path}")

if __name__ == "__main__":
    # usage: python src/preprocess.py config/preprocess_config.yaml
    config_file = sys.argv[1]
    main(config_file)
