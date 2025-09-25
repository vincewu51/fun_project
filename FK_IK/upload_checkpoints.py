from huggingface_hub import upload_folder

upload_folder(
    repo_id="siyulw2025/trash_picking_cache",
    folder_path="outputs/train/minions001/checkpoints",
    path_in_repo="trash_picking_cache",
    commit_message="Upload all checkpoints from 20k to 100k"
)