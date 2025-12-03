"""
Robust training + tuning + MLflow logging + Hugging Face model registration script
Patched for sklearn OneHotEncoder compatibility, CV-results saving, and MLflow URI fallback.

This file:
- downloads processed Xtrain/Xtest/ytrain/ytest from the HF dataset repo (processed/)
- trains multiple classical models (DecisionTree, Bagging, RandomForest, AdaBoost, GradientBoosting, XGBoost if available)
- runs GridSearchCV for each candidate, logs best params and CV results to MLflow (and saves cv_results_ JSON)
- evaluates on test set, logs test metrics to MLflow, stores artifacts locally
- registers the best model in the Hugging Face Model Hub (creates repo if needed)
"""
# for data manipulation & printing
import os
import joblib
import json
import tempfile
import shutil
from pprint import pprint

import pandas as pd

# sklearn preprocessing & pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

# OneHotEncoder compatibility: import alias and create wrapper later
from sklearn.preprocessing import OneHotEncoder as _OHE

# for model training, tuning, and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# optional xgboost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# huggingface hub helpers
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# mlflow
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------
# Config - edit these values before running
# -----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Please set HF_TOKEN environment variable before running.")

# dataset repo where processed train/test CSVs live (we expect processed/Xtrain.csv etc.)
HF_DATASET_REPO = "sathishaiuse/Tourism-Package"
PROCESSED_PATH = "processed"
XTRAIN_FILE = f"{PROCESSED_PATH}/Xtrain.csv"
XTEST_FILE = f"{PROCESSED_PATH}/Xtest.csv"
YTRAIN_FILE = f"{PROCESSED_PATH}/ytrain.csv"
YTEST_FILE = f"{PROCESSED_PATH}/ytest.csv"

# HF model repo to register best model
HF_MODEL_REPO = "sathishaiuse/wellness-classifier-model"

# Local working dirs
WORK_DIR = "tourism_project/model_building/work"
ARTIFACT_DIR = "tourism_project/model_building/artifacts"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# MLflow configuration (fallback to file backend if server unreachable)
DEFAULT_MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", "http://localhost:5000")
try:
    mlflow.set_tracking_uri(DEFAULT_MLFLOW_SERVER)
    # basic connectivity check using MlflowClient (this will not always raise, but is a best-effort)
    client = MlflowClient(tracking_uri=DEFAULT_MLFLOW_SERVER)
    _ = client.list_experiments()  # may raise if server not reachable
    print("Using MLflow tracking server at:", DEFAULT_MLFLOW_SERVER)
except Exception:
    fallback = "file:///tmp/mlruns"
    mlflow.set_tracking_uri(fallback)
    print("Could not reach MLflow server; using file-backed MLflow tracking at:", fallback)

mlflow.set_experiment("mlops-training-experiment")

# Modeling choices
TARGET = "ProdTaken"   # target column in your processed CSVs (0/1)
TEST_THRESHOLD = 0.5
CV = 3
RANDOM_STATE = 42

# -----------------------
# Helper functions
# -----------------------
api = HfApi(token=HF_TOKEN)

def download_from_dataset_repo(filename_in_repo: str, local_dir: str) -> str:
    """
    Download a file from the HF dataset repo and copy it into local_dir.
    Accepts either a basename (e.g., 'Xtrain.csv') or a repo-relative path (e.g., 'processed/Xtrain.csv').
    Returns the local path to the copied file.
    """
    # if given a basename, prepend PROCESSED_PATH
    if "/" not in filename_in_repo:
        remote_filename = f"{PROCESSED_PATH}/{filename_in_repo}"
    else:
        remote_filename = filename_in_repo

    cache_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=remote_filename,
        repo_type="dataset",
        use_auth_token=HF_TOKEN
    )
    dest = os.path.join(local_dir, os.path.basename(cache_path))
    if cache_path != dest:
        shutil.copy(cache_path, dest)
    return dest

def safe_create_model_repo(repo_id: str):
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Model repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{repo_id}' ...")
        create_repo(repo_id=repo_id, repo_type="model", private=False, token=HF_TOKEN)
        print("Created model repo.")
    except HfHubHTTPError as e:
        raise SystemExit(f"Failed checking/creating model repo: {e}")

# -----------------------
# Download processed CSVs from HF dataset repo
# -----------------------
print("Downloading processed CSVs from Hugging Face dataset repo...")
local_xtrain = download_from_dataset_repo(os.path.basename(XTRAIN_FILE), WORK_DIR)
local_xtest  = download_from_dataset_repo(os.path.basename(XTEST_FILE), WORK_DIR)
local_ytrain = download_from_dataset_repo(os.path.basename(YTRAIN_FILE), WORK_DIR)
local_ytest  = download_from_dataset_repo(os.path.basename(YTEST_FILE), WORK_DIR)
print("Downloaded files:")
print(local_xtrain, local_xtest, local_ytrain, local_ytest)

# read into pandas
Xtrain = pd.read_csv(local_xtrain)
Xtest  = pd.read_csv(local_xtest)
ytrain = pd.read_csv(local_ytrain)
ytest  = pd.read_csv(local_ytest)

# flatten y to Series if single-column
if ytrain.shape[1] == 1:
    ytrain_s = ytrain.iloc[:, 0]
else:
    ytrain_s = ytrain.iloc[:, 0]

if ytest.shape[1] == 1:
    ytest_s = ytest.iloc[:, 0]
else:
    ytest_s = ytest.iloc[:, 0]

print("Shapes -> Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape, "ytrain:", ytrain_s.shape, "ytest:", ytest_s.shape)

# -----------------------
# Feature lists (from your problem statement)
# -----------------------
numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

# select only columns that exist (defensive)
numeric_features = [c for c in numeric_features if c in Xtrain.columns]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

print("Numeric features used:", numeric_features)
print("Categorical features used:", categorical_features)

# -----------------------
# Preprocessor and pipeline pattern (OneHotEncoder compatibility)
# -----------------------
# Build a OneHotEncoder instance that works across sklearn versions
try:
    ohe = _OHE(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = _OHE(handle_unknown="ignore", sparse=False)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (ohe, categorical_features),
    remainder="drop"
)

# We'll use a named step 'clf' so parameter grids can target clf__*
def make_model_pipeline(estimator):
    return Pipeline([("preproc", preprocessor), ("clf", estimator)])

# -----------------------
# Candidate models and small grids (keeps runtime reasonable)
# -----------------------
models_and_grids = {
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5]
        }
    },
    "Bagging": {
        "estimator": BaggingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [10, 50],
            "clf__max_samples": [0.5, 1.0]
        }
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [None, 10]
        }
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.5, 1.0]
        }
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.1, 0.05]
        }
    }
}

if HAS_XGB:
    # compute scale_pos_weight for xgboost from training set labels (if binary)
    vc = ytrain_s.value_counts()
    scale_pos = float(vc.iloc[0]) / float(vc.iloc[1]) if len(vc) > 1 else 1.0
    models_and_grids["XGBoost"] = {
        "estimator": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1, scale_pos_weight=scale_pos),
        "param_grid": {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [3, 6],
            "clf__learning_rate": [0.1, 0.05]
        }
    }
else:
    print("XGBoost not available; skipping XGBoost model.")

# -----------------------
# Grid search, MLflow logging, evaluation
# -----------------------
results = {}
best_model_info = {"name": None, "model": None, "f1": -1.0, "local_path": None}

for name, mg in models_and_grids.items():
    estimator = mg["estimator"]
    param_grid = mg["param_grid"]

    print("\n" + "="*60)
    print(f"Training & tuning: {name}")
    pipeline = make_model_pipeline(estimator)

    # Grid search
    gs = GridSearchCV(pipeline, param_grid, cv=CV, scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(Xtrain, ytrain_s)

    print(f"Best params for {name}:")
    pprint(gs.best_params_)

    # -----------------------
    # Save full cv_results_ for auditability (json)
    # -----------------------
    cv_results_fp = os.path.join(ARTIFACT_DIR, f"cv_results_{name}.json")
    cv_results_clean = {}
    for k, v in gs.cv_results_.items():
        try:
            cv_results_clean[k] = v.tolist()
        except Exception:
            cv_results_clean[k] = v
    with open(cv_results_fp, "w") as fh:
        json.dump(cv_results_clean, fh, indent=2)

    # Evaluate best on test
    best_est = gs.best_estimator_
    if hasattr(best_est, "predict_proba"):
        ypred_proba = best_est.predict_proba(Xtest)[:, 1]
        ypred = (ypred_proba >= TEST_THRESHOLD).astype(int)
    else:
        ypred = best_est.predict(Xtest)

    try:
        acc = accuracy_score(ytest_s, ypred)
        prec = precision_score(ytest_s, ypred, zero_division=0)
        rec = recall_score(ytest_s, ypred, zero_division=0)
        f1 = f1_score(ytest_s, ypred, zero_division=0)
    except Exception:
        # fallback for multiclass
        prec = precision_score(ytest_s, ypred, average="weighted", zero_division=0)
        rec = recall_score(ytest_s, ypred, average="weighted", zero_division=0)
        f1 = f1_score(ytest_s, ypred, average="weighted", zero_division=0)
        acc = accuracy_score(ytest_s, ypred)

    print(f"Test metrics for {name}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

    # Log to MLflow - main run for this model
    with mlflow.start_run(run_name=f"{name}_gridsearch"):
        # Log best params and CV score
        mlflow.log_params(gs.best_params_)
        if hasattr(gs, "best_score_"):
            mlflow.log_metric("cv_best_score", float(gs.best_score_))

        # Log test metrics
        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_precision", float(prec))
        mlflow.log_metric("test_recall", float(rec))
        mlflow.log_metric("test_f1", float(f1))

        # Attach cv_results_ JSON as artifact so all tuned parameters are preserved
        try:
            mlflow.log_artifact(cv_results_fp, artifact_path=f"cv_results/{name}")
        except Exception:
            # if MLflow server not reachable, file is still present in ARTIFACT_DIR
            pass

        # Log each param combo from cv_results_ as nested runs (optional)
        cv_results = gs.cv_results_
        for i in range(len(cv_results["params"])):
            params_i = cv_results["params"][i]
            mean_test = float(cv_results["mean_test_score"][i]) if "mean_test_score" in cv_results else None
            std_test = float(cv_results["std_test_score"][i]) if "std_test_score" in cv_results else None
            with mlflow.start_run(nested=True):
                if params_i:
                    mlflow.log_params(params_i)
                if mean_test is not None:
                    mlflow.log_metric("mean_test_score", mean_test)
                if std_test is not None:
                    mlflow.log_metric("std_test_score", std_test)

        # Save model locally and log as artifact
        model_fp = os.path.join(ARTIFACT_DIR, f"best_{name}.joblib")
        joblib.dump(best_est, model_fp)
        try:
            mlflow.log_artifact(model_fp, artifact_path="models")
        except Exception:
            pass

        # also save a small JSON with test metrics
        metrics_fp = os.path.join(ARTIFACT_DIR, f"metrics_{name}.json")
        with open(metrics_fp, "w") as fh:
            json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, fh, indent=2)
        try:
            mlflow.log_artifact(metrics_fp, artifact_path="metrics")
        except Exception:
            pass

    # store results
    results[name] = {
        "best_params": gs.best_params_,
        "cv_best_score": float(gs.best_score_) if hasattr(gs, "best_score_") else None,
        "test_metrics": {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    }

    # track best overall by test f1
    if f1 > best_model_info["f1"]:
        best_model_info["name"] = name
        best_model_info["model"] = best_est
        best_model_info["f1"] = f1
        best_local_path = os.path.join(ARTIFACT_DIR, f"best_overall_{name}.joblib")
        joblib.dump(best_est, best_local_path)
        best_model_info["local_path"] = best_local_path

# -----------------------
# Save tuning summary locally
# -----------------------
results_fp = os.path.join(ARTIFACT_DIR, "tuning_results_summary.json")
with open(results_fp, "w") as fh:
    json.dump(results, fh, indent=2)
print("Saved tuning summary to", results_fp)

# -----------------------
# Register best model in HF Model Hub
# -----------------------
if best_model_info["model"] is None:
    raise SystemExit("No model trained successfully.")

print("Best model:", best_model_info["name"], "F1:", best_model_info["f1"])
safe_create_model_repo(HF_MODEL_REPO)

# Upload model pickle and metrics to the model repo
upload_items = {
    best_model_info["local_path"]: os.path.basename(best_model_info["local_path"]),
    results_fp: os.path.basename(results_fp)
}
# include per-model metric files in ARTIFACT_DIR
for f in os.listdir(ARTIFACT_DIR):
    if f.startswith("metrics_") or f.endswith(".joblib") or f.startswith("cv_results_"):
        fp = os.path.join(ARTIFACT_DIR, f)
        upload_items[fp] = f

for local_fp, remote_name in upload_items.items():
    print(f"Uploading {local_fp} -> {HF_MODEL_REPO}/{remote_name}")
    api.upload_file(
        path_or_fileobj=local_fp,
        path_in_repo=remote_name,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        token=HF_TOKEN
    )

print("Model registration to Hugging Face Model Hub completed.")
print(f"View model repo: https://huggingface.co/{HF_MODEL_REPO}")
