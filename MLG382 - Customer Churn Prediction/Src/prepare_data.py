# src/prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_clean_data(filepath):
    # Load the raw dataset
    df = pd.read_csv(filepath)

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Replace empty strings in 'TotalCharges' with NaN and drop them
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)

    return df

def split_and_save_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['Churn'], random_state=random_state)

    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print(f"✅ Saved {len(train_df)} rows to data/train.csv")
    print(f"✅ Saved {len(test_df)} rows to data/test.csv")

if __name__ == "__main__":
    raw_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv" 
    data = load_and_clean_data(raw_path)
    split_and_save_data(data)
