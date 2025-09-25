from huggingface_hub import upload_folder

upload_folder(
    repo_id="siyulw2025/trash_picking_cache",
    folder_path="outputs/train/minions001/checkpoints",
    path_in_repo="trash_picking_cache",
    commit_message="Upload all checkpoints from 20k to 100k"
)

dataset_root = "meta/episodes"
repo_id = "siyulw2025/trash_picking_cache"

for root, _, files in os.walk(dataset_root):
    for fname in files:
        if fname.endswith(".parquet"):
            local_path = os.path.join(root, fname)
            
            # Mirror same folder structure inside the repo
            relative_path = os.path.relpath(local_path, dataset_root)
            repo_path = f"datasets/{relative_path}"
            
            print(f"Uploading {local_path} → {repo_path}")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                commit_message=f"Upload dataset file {relative_path}"
            )
