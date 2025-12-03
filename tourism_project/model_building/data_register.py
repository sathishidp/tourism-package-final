from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "sathishaiuse/Tourism-Package"
repo_type = "dataset"
data_folder = "tourism_project/data"

# Ensure HF token is available
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please export your Hugging Face token.")

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using existing repo.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repository '{repo_id}' created successfully.")

except HfHubHTTPError as e:
    print(f"Error checking repository: {e}")
    raise

# ----------------------------------------------------
# Upload folder to HF dataset repo
# ----------------------------------------------------
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Data folder not found: {data_folder}")

print(f"Uploading dataset from '{data_folder}' to Hugging Face Hub...")
api.upload_folder(
    folder_path=data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Initial dataset upload from MLOps pipeline"
)

print("Upload completed successfully!")
