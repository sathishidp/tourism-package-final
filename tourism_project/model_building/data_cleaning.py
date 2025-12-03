"""
Data cleaning & preprocessing for the Tourism Package dataset.

This script:
- Downloads the raw CSV (tourism.csv) from a Hugging Face dataset repo
- Performs lightweight cleaning aligned with your EDA:
    * drops index/ID columns
    * removes duplicates
    * ensures correct dtypes
    * clips outliers for DurationOfPitch
    * converts categorical columns to 'category'
- Saves a cleaned full CSV and stratified train/test splits locally:
    tourism_project/data/processed/cleaned_tourism.csv
    tourism_project/data/processed/Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv
- Uploads the processed files back to the same Hugging Face dataset repo under 'processed/'

Usage:
    export HF_TOKEN="hf_xxx..."
    python tourism_project/model_building/data_cleaning.py \
        --repo-id sathishaiuse/Tourism-Package \
        --raw-filename tourism.csv \
        --processed-path tourism_project/data/processed \
        --test-size 0.2 \
        --random-state 42
"""
import os
import argparse
import tempfile
import shutil
from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# -------------------------
# Default config
# -------------------------
DEFAULT_REPO_ID = "sathishaiuse/Tourism-Package"
DEFAULT_RAW_FILENAME = "tourism.csv"
DEFAULT_PROCESSED_PATH = "tourism_project/data/processed"
TARGET_COL = "ProdTaken"   # expected target
# features (from problem statement)
NUMERIC_FEATURES = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation", "ProductPitched"
]

# -------------------------
# Helper functions
# -------------------------
def safe_cast_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id (username/repo)")
    parser.add_argument("--raw-filename", default=DEFAULT_RAW_FILENAME, help="Raw CSV filename inside the dataset repo")
    parser.add_argument("--processed-path", default=DEFAULT_PROCESSED_PATH, help="Local folder to save processed files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable name for HF token")
    args = parser.parse_args()

    hf_token = os.getenv(args.hf_token_env)
    if not hf_token:
        raise SystemExit(f"HF token not found in environment variable {args.hf_token_env}. Set it first and retry.")

    repo_id = args.repo_id
    raw_filename = args.raw_filename
    processed_path = args.processed_path
    test_size = args.test_size
    random_state = args.random_state

    os.makedirs(processed_path, exist_ok=True)

    api = HfApi(token=hf_token)

    # Download raw CSV from HF dataset repo
    print(f"Downloading {raw_filename} from Hugging Face dataset {repo_id} ...")
    try:
        cache_path = hf_hub_download(repo_id=repo_id, filename=raw_filename, repo_type="dataset", use_auth_token=hf_token)
    except Exception as e:
        raise SystemExit(f"Failed to download {raw_filename} from {repo_id}: {e}")

    print("Loaded raw file from cache:", cache_path)
    df = pd.read_csv(cache_path)
    print("Raw dataframe shape:", df.shape)

    # -------------------------
    # Cleaning steps
    # -------------------------
    # 1) Drop index/ID columns (if present)
    drop_cols = []
    for c in ["Unnamed: 0", "CustomerID"]:
        if c in df.columns:
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print("Dropped columns:", drop_cols)

    # 2) Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Removed duplicates: {before - after} rows")

    # 3) Cast numeric features
    df = safe_cast_numeric(df, NUMERIC_FEATURES + [TARGET_COL])

    # 4) Clip DurationOfPitch to reasonable max (e.g., 60 minutes) to reduce extreme outliers
    if "DurationOfPitch" in df.columns:
        max_clip = 60
        n_outliers = (df["DurationOfPitch"] > max_clip).sum()
        if n_outliers > 0:
            print(f"Clipping {n_outliers} DurationOfPitch values to {max_clip}")
            df["DurationOfPitch"] = df["DurationOfPitch"].clip(upper=max_clip)

    # 5) Ensure categorical dtypes
    df = ensure_categorical(df, CATEGORICAL_FEATURES)

    # 6) Fill or report any remaining NA (should be none per EDA)
    na_summary = df.isna().sum()
    na_total = na_summary.sum()
    if na_total > 0:
        print("Found missing values (per column):")
        print(na_summary[na_summary > 0])
        # Basic imputation: numeric -> median, categorical -> mode
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
                    print(f"Filled NA in numeric {col} with median={med}")
                else:
                    try:
                        mode = df[col].mode(dropna=True)[0]
                    except Exception:
                        mode = ""
                    df[col] = df[col].fillna(mode)
                    print(f"Filled NA in categorical {col} with mode='{mode}'")
    else:
        print("No missing values detected.")

    # Re-check types & summary
    print("\nPost-cleaning shape and dtypes:")
    print(df.shape)
    print(df.dtypes.value_counts())
    print(df.head(3))

    # Save cleaned full CSV
    cleaned_fp = os.path.join(processed_path, "cleaned_tourism.csv")
    df.to_csv(cleaned_fp, index=False)
    print("Saved cleaned dataset to:", cleaned_fp)

    # -------------------------
    # Train/test split (stratified)
    # -------------------------
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column '{TARGET_COL}' not found in cleaned dataframe. Aborting split.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Stratify to preserve class imbalance ratio in splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save splits to processed_path
    Xtrain_fp = os.path.join(processed_path, "Xtrain.csv")
    Xtest_fp  = os.path.join(processed_path, "Xtest.csv")
    ytrain_fp = os.path.join(processed_path, "ytrain.csv")
    ytest_fp  = os.path.join(processed_path, "ytest.csv")

    X_train.to_csv(Xtrain_fp, index=False)
    X_test.to_csv(Xtest_fp, index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(ytrain_fp, index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(ytest_fp, index=False)

    print("Saved train/test splits to:", processed_path)
    print(" ->", Xtrain_fp)
    print(" ->", Xtest_fp)
    print(" ->", ytrain_fp)
    print(" ->", ytest_fp)

    # -------------------------
    # Upload processed files back to HF dataset repo under 'processed/'
    # -------------------------
    tmp_upload_dir = tempfile.mkdtemp(prefix="hf_processed_")
    try:
        shutil.copy(cleaned_fp, os.path.join(tmp_upload_dir, "cleaned_tourism.csv"))
        shutil.copy(Xtrain_fp, os.path.join(tmp_upload_dir, "Xtrain.csv"))
        shutil.copy(Xtest_fp, os.path.join(tmp_upload_dir, "Xtest.csv"))
        shutil.copy(ytrain_fp, os.path.join(tmp_upload_dir, "ytrain.csv"))
        shutil.copy(ytest_fp, os.path.join(tmp_upload_dir, "ytest.csv"))

        print("Uploading processed files to Hugging Face dataset repo under 'processed/' ...")
        api.upload_folder(
            folder_path=tmp_upload_dir,
            path_in_repo="processed",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            commit_message="Add cleaned dataset and train/test splits from data_cleaning.py"
        )
        print("Upload complete. Processed files are in the dataset repo under 'processed/'")
    except Exception as e:
        print("Upload failed:", e)
        raise
    finally:
        shutil.rmtree(tmp_upload_dir, ignore_errors=True)

    # Print a short summary of label balance
    print("\nLabel distribution (full dataset):")
    print(y.value_counts(normalize=False))
    print("\nLabel distribution (train):")
    print(y_train.value_counts(normalize=False))
    print("\nLabel distribution (test):")
    print(y_test.value_counts(normalize=False))

if __name__ == "__main__":
    main()
