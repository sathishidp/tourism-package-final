
# for data manipulation
import pandas as pd
# for creating folders
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# huggingface helpers to download & upload files
from huggingface_hub import HfApi, hf_hub_download

# -----------------------
# Config - change if needed
# -----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Please set HF_TOKEN environment variable before running.")

api = HfApi(token=HF_TOKEN)

# Hugging Face dataset repo + filename inside it
HF_REPO_ID = "sathishaiuse/Tourism-Package"   # Hugging Face dataset repo
HF_FILENAME = "tourism.csv"                   # the CSV file inside the repo

# local folder to save processed files
LOCAL_PROCESSED_DIR = "tourism_project/data/processed"
os.makedirs(LOCAL_PROCESSED_DIR, exist_ok=True)

# -----------------------
# Download dataset CSV from HF Hub
# -----------------------
print("Downloading dataset file from Hugging Face Hub...")
local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset", use_auth_token=HF_TOKEN)
print("Downloaded to:", local_path)

# Load into pandas
df = pd.read_csv(local_path)
print("Dataset loaded successfully. Shape:", df.shape)

# -----------------------
# Define target and features per your problem statement
# -----------------------
# Target variable from the Data Dictionary
target = "ProdTaken"   # 0: No, 1: Yes

# We'll drop CustomerID from features (identifier) and keep other relevant columns.
# Numeric features (as per Data Dictionary)
numeric_features = [
    "Age",
    "CityTier",                    # Tier 1/2/3 encoded as numbers in dataset
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",                    # 0/1
    "OwnCar",                      # 0/1
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

# Categorical / text features (as per Data Dictionary)
categorical_features = [
    "TypeofContact",               # Company Invited / Self Inquiry
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

# Some datasets include CustomerID; ensure we don't include it as a feature
exclude_cols = ["CustomerID"]

# Build feature matrix X and target y
# Keep only columns that exist in the loaded df to avoid crashes if something is missing
selected_numeric = [c for c in numeric_features if c in df.columns]
selected_categorical = [c for c in categorical_features if c in df.columns]
selected_features = [c for c in selected_numeric + selected_categorical if c not in exclude_cols]

if target not in df.columns:
    raise SystemExit(f"Target column '{target}' not found in dataset. Available columns: {list(df.columns)}")

X = df[selected_features]
y = df[[target]]   # keep as DataFrame so saving produces a CSV with header

print("Using features:", selected_features)
print("Target:", target)

# -----------------------
# Simple preprocessing (optional, minimal so it looks like your original work)
# - Fill NA in numeric cols with median, categorical with mode (keeps it simple & reproducible)
# -----------------------
for col in selected_numeric:
    if col in X.columns:
        if X[col].isnull().any():
            med = X[col].median()
            X[col] = X[col].fillna(med)
            print(f"Filled NA in numeric column '{col}' with median={med}")

for col in selected_categorical:
    if col in X.columns:
        if X[col].isnull().any():
            try:
                m = X[col].mode(dropna=True)[0]
            except IndexError:
                m = ""
            X[col] = X[col].fillna(m)
            print(f"Filled NA in categorical column '{col}' with mode='{m}'")

# -----------------------
# Train/test split
# -----------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save locally
Xtrain_fp = os.path.join(LOCAL_PROCESSED_DIR, "Xtrain.csv")
Xtest_fp  = os.path.join(LOCAL_PROCESSED_DIR, "Xtest.csv")
ytrain_fp = os.path.join(LOCAL_PROCESSED_DIR, "ytrain.csv")
ytest_fp  = os.path.join(LOCAL_PROCESSED_DIR, "ytest.csv")

Xtrain.to_csv(Xtrain_fp, index=False)
Xtest.to_csv(Xtest_fp, index=False)
ytrain.to_csv(ytrain_fp, index=False)
ytest.to_csv(ytest_fp, index=False)

print("Saved processed files locally:")
print(Xtrain_fp, Xtest_fp, ytrain_fp, ytest_fp)

# -----------------------
# Upload processed files back to HF dataset repo under 'processed/' folder
# -----------------------
files = [Xtrain_fp, Xtest_fp, ytrain_fp, ytest_fp]
for file_path in files:
    file_name_only = os.path.basename(file_path)
    remote_path = f"processed/{file_name_only}"   # place under 'processed/' in the dataset repo
    print(f"Uploading {file_path} -> {HF_REPO_ID}:{remote_path}")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=remote_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

print("All files uploaded successfully.")
