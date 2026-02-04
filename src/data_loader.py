"""Data loading utilities for VI Labs assignment."""

import os
import pandas as pd


def load_csvs(data_dir="data/train/"):
    """
    Load training or test data from CSV files.
    
    Args:
        data_dir: Directory containing the CSV files (default: "data/train/")
        
    Returns:
        tuple: (churn_df, app_df, web_df, claims_df)
    """
    if os.path.exists(os.path.join(data_dir, "churn_labels.csv")):
        base = data_dir
        print(f"[LOAD] Loading from folder: {base}")
        churn  = pd.read_csv(os.path.join(base, "churn_labels.csv"), parse_dates=["signup_date"])
        app    = pd.read_csv(os.path.join(base, "app_usage.csv"), parse_dates=["timestamp"])
        web    = pd.read_csv(os.path.join(base, "web_visits.csv"), parse_dates=["timestamp"])
        claims = pd.read_csv(os.path.join(base, "claims.csv"), parse_dates=["diagnosis_date"])
    else:
        print("[LOAD] Loading from uploaded files in /mnt/data")
        churn  = pd.read_csv("/mnt/data/churn_labels.csv", parse_dates=["signup_date"])
        app    = pd.read_csv("/mnt/data/app_usage.csv", parse_dates=["timestamp"])
        web    = pd.read_csv("/mnt/data/web_visits.csv", parse_dates=["timestamp"])
        claims = pd.read_csv("/mnt/data/claims.csv", parse_dates=["diagnosis_date"])
    return churn, app, web, claims

