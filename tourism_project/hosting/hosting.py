"""
Robust uploader to create (if missing) and push a deployment folder to a Hugging Face Space.

Usage in CI:
  - Export HF_TOKEN and HF_SPACE_ID as secrets.
  - Optionally set SPACE_SDK (docker|gradio|streamlit|static) and SPACE_PRIVATE (true/false).
  - Ensure LOCAL_DEPLOY_FOLDER points to your deployment files (Dockerfile/app.py/requirements.txt).
"""
import os
import sys
import traceback
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# -----------------------
# Config (read from env / CI secrets)
# -----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Add HF_TOKEN as a GitHub secret and export it.", file=sys.stderr)
    sys.exit(1)

SPACE_REPO_ID = os.getenv("HF_SPACE_ID", "").strip()  # must be like 'username/space-name'
if not SPACE_REPO_ID:
    print("ERROR: HF_SPACE_ID not set. Add HF_SPACE_ID as a GitHub secret in the form 'username/space-name'.", file=sys.stderr)
    sys.exit(1)

SPACE_SDK = os.getenv("SPACE_SDK", "docker").lower()
if SPACE_SDK not in ("docker", "gradio", "streamlit", "static"):
    print(f"ERROR: Invalid SPACE_SDK='{SPACE_SDK}'. Choose docker, gradio, streamlit or static.", file=sys.stderr)
    sys.exit(1)

PRIVATE = os.getenv("SPACE_PRIVATE", "false").lower() in ("1", "true", "yes")
LOCAL_DEPLOY_FOLDER = os.getenv("LOCAL_DEPLOY_FOLDER", "tourism_project/deployment")

# -----------------------
# Basic validation of repo id format
# -----------------------
if "/" not in SPACE_REPO_ID or SPACE_REPO_ID.count("/") != 1:
    print("ERROR: HF_SPACE_ID must be 'username/space-name' (single slash). Got:", SPACE_REPO_ID, file=sys.stderr)
    sys.exit(1)

# -----------------------
# Validate deployment folder
# -----------------------
if not os.path.isdir(LOCAL_DEPLOY_FOLDER):
    print(f"ERROR: Local deployment folder '{LOCAL_DEPLOY_FOLDER}' not found. Add Dockerfile/app.py/requirements.txt.", file=sys.stderr)
    sys.exit(1)

# -----------------------
# Initialize API
# -----------------------
api = HfApi(token=HF_TOKEN)

# -----------------------
# Ensure space exists (create if not)
# -----------------------
try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space")
    print(f"Space '{SPACE_REPO_ID}' exists. Proceeding to upload.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO_ID}' not found — creating with SDK='{SPACE_SDK}' (private={PRIVATE}) ...")
    try:
        # Use create_repo with repo_id and repo_type for this hf version
        api.create_repo(repo_id=SPACE_REPO_ID, repo_type="space", private=PRIVATE, space_sdk=SPACE_SDK, token=HF_TOKEN)
        print(f"Successfully created space '{SPACE_REPO_ID}'.")
    except TypeError:
        # Older/newer hf versions might not accept space_sdk or repo_id param names; try alternate signatures
        try:
            # try legacy call: create_repo(name, repo_type=..., organisation=..., private=...)
            owner, name = SPACE_REPO_ID.split("/")
            api.create_repo(name=name, repo_type="space", organization=owner, private=PRIVATE, space_sdk=SPACE_SDK, token=HF_TOKEN)
            print(f"Created space via alternate create_repo signature: {SPACE_REPO_ID}")
        except Exception as e2:
            print("Failed to create space (alternate attempt). Error:", e2, file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print("Failed to create space. Error:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
except HfHubHTTPError as e:
    print("HTTP error when checking space info:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print("Unexpected error while checking/creating space:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# -----------------------
# Upload deployment folder
# -----------------------
print(f"Uploading '{LOCAL_DEPLOY_FOLDER}' to space '{SPACE_REPO_ID}' ...")
try:
    upload_folder(
        folder_path=LOCAL_DEPLOY_FOLDER,
        path_in_repo=".",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
        commit_message="Upload deployment files from CI"
    )
    print("Upload finished successfully.")
    print(f"Visit the Space: https://huggingface.co/spaces/{SPACE_REPO_ID}")
except Exception as e:
    print("Upload failed:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
