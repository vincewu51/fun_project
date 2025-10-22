from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()

repo_id = "siyulw2025/xlerobot-candybar-rightarm-002"
local_path = Path("/Users/siyulw/workspace/xlerobot-data/xlerobot-candybar-rightarm-002/")

# Make sure the repo exists (create if needed)
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"✅ Repo ready: {repo_id}")
except Exception as e:
    print(f"⚠️ Repo creation skipped or failed: {e}")

# Upload recursively
api.upload_folder(
    folder_path=str(local_path),
    repo_id=repo_id,
    repo_type="dataset",
)
print(f"✅ Uploaded {local_path} → {repo_id}")

