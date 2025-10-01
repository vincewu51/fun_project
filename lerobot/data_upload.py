from huggingface_hub import upload_folder, HfApi

# ---- Configuration ----
local_dataset_path = "/media/yifeng-wu/E/EverNorif/so101_test_orange_pick_v033"
repo_id = "siyulw2025/so101_test_orange_pick_gr00t"
branch = "v2.1"
dataset_tag = "LeRobot"

# ---- Step 1: Make sure repo exists ----
api = HfApi()
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True,
)

if branch:
    api.create_branch(
        repo_id=repo_id,
        branch=branch,
        repo_type="dataset",
        exist_ok=True,
    )

# ---- Step 2: Upload the dataset folder to branch 'v3.0' ----
upload_folder(
    folder_path=local_dataset_path,
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="",             # Upload to root of dataset
    #revision=branch,             # Upload to the 'v3.0' branch
)

# ---- Step 3: Add the 'lerobot' tag ----
#api.create_tag(repo_id, tag=branch, revision=branch, repo_type="dataset")

print(f"✅ Dataset pushed to https://huggingface.co/datasets/{repo_id}/tree/{branch}")
