# data_loader.py
#KDD cup 1999 cybersecurity dataset

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder

def load_kdd_data():
    print("Loading KDD cup 1999 dataset...")
    print("(First run downlaods ~18MB automatically - be patient)\n")


    raw = fetch_kddcup99(
        subset=None,
        as_frame=True,
        percent10=True,
        random_state=42
    )

    df = raw.frame.copy()

    print(f"Dataset Loaded Successfully!")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    return df

def inspect_data(df):
    print("-- FIRST 5 ROWS --")
    print(df.head())
    print()

    print("-- COLUMN DATA TYPES --")
    print(df.dtypes)
    print()

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("-- MISSING VALUES --")
        print(" No Missing Values Found!")
    else:
        print("-- MISSING VALUES FOUND --")
        print(missing)
        print()

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("--CATEGORICAL (TEXT) COLUMNS--")
    print(categorical_cols)
    print()

    return categorical_cols

def encode_and_prepare(df, categorical_cols):
    

    df = df.copy()  

    
    target_col = 'labels'

   
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' not found. "
                          f"Available columns: {df.columns.tolist()}")

    
    df[target_col] = df[target_col].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
    )
    df[target_col] = df[target_col].str.strip('.')
    df['binary_target'] = (df[target_col] != 'normal').astype(int)

    encoders = {}
    for col in categorical_cols:
        if col == target_col:
            continue  
        le = LabelEncoder()
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    df = df.drop(columns=[target_col])

    return df, encoders


def check_imbalance(df, target_col='binary_target'):
    counts = df[target_col].value_counts()


    normal_count = counts.get(0, 0)
    attack_count = counts.get(1, 0)
    total = normal_count + attack_count

    print("-- CLASS DISTRIBUTION --")
    print(f"Normal (0): {normal_count:,} rows ({normal_count/total:2%})")
    print(f"Attack (1): {attack_count:,} rows ({attack_count/total:.2%})")

    if attack_count == 0:
        print("\n WARNING NO ATTACK SAMPLES FOUND.")
        return None
    
    ratio = normal_count / attack_count
    print(f"\nImbalance Ratio -> {ratio:.1f} : 1 (normal : attack)")
    print()

    return ratio



def get_clean_data():
  
    df = load_kdd_data()
    categorical_cols = inspect_data(df)
    df, encoders = encode_and_prepare(df, categorical_cols)
    check_imbalance(df, target_col='binary_target')

    
    X = df.drop(columns=['binary_target'])
    y = df['binary_target']

    return X, y



if __name__ == "__main__":
    X, y = get_clean_data()
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
