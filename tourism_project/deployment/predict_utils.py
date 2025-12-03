import os
import joblib
import shutil
from huggingface_hub import hf_hub_download, HfApi
from typing import List

def download_model_from_hf(model_repo: str, model_filename: str = None, token: str = None, local_dir: str = "/app/model"):
    """
    Try to download the model file from HF model repo.
    If model_filename is None, attempt fallback names (best_overall_XGBoost, RandomForest, Bagging, DecisionTree).
    Returns local filepath.
    """
    os.makedirs(local_dir, exist_ok=True)
    api = HfApi(token=token)

    candidates = []
    if model_filename:
        candidates.append(model_filename)

    # fallback candidates (order of preference)
    candidates.extend([
        "best_overall_XGBoost.joblib",
        "best_overall_RandomForest.joblib",
        "best_overall_Bagging.joblib",
        "best_overall_DecisionTree.joblib",
        "best_XGBoost.joblib",
        "best_RandomForest.joblib",
        "best_Bagging.joblib",
        "best_DecisionTree.joblib",
    ])

    last_exception = None
    for fn in candidates:
        try:
            print(f"Trying to download '{fn}' from '{model_repo}' ...")
            remote = hf_hub_download(repo_id=model_repo, filename=fn, repo_type="model", use_auth_token=token)
            # hf_hub_download returns a cache path; copy into local_dir with same filename
            dest = os.path.join(local_dir, os.path.basename(remote))
            if remote != dest:
                shutil.copy(remote, dest)
            print("Downloaded model to:", dest)
            return dest
        except Exception as e:
            last_exception = e
            print(f"Could not download {fn}: {e}")

    # If we got here no candidate succeeded
    raise FileNotFoundError(f"Model not found in repo '{model_repo}'. Tried: {candidates}. Last error: {last_exception}")

def load_model(local_model_path: str):
    """Load joblib model/pipeline from given local path."""
    return joblib.load(local_model_path)

def inputs_to_dataframe(payload: dict, feature_order: List[str]):
    """
    Convert one record (dict) to dataframe with fixed column order.
    """
    import pandas as pd
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Payload must be dict or list of dicts")

    df = pd.DataFrame(rows)
    # ensure columns exist
    for c in feature_order:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[feature_order].copy()
    # try cast numeric columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass
    return df
