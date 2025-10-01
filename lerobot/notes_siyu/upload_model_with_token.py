#!/usr/bin/env python3

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_model_to_hf():
    """Upload the ACT model to Hugging Face Hub"""

    # Configuration
    model_name = "siyulw2025/gr00t_orange"
    model_path = "/workspace/Isaac-GR00T/so101-decord-checkpoints/"

    # Check for token in environment variable
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ Please set your Hugging Face token as an environment variable:")
        print("export HF_TOKEN='your_token_here'")
        print("\nYou can get a token from: https://huggingface.co/settings/tokens")
        return

    # Initialize the API with token
    api = HfApi(token=hf_token)

    try:
        # Create the repository (this will fail gracefully if it already exists)
        print(f"Creating repository: {model_name}")
        create_repo(
            repo_id=model_name,
            repo_type="model",
            exist_ok=True,
            private=False,
            token=hf_token
        )
        print(f"Repository {model_name} is ready!")

        # Upload all files from the model directory
        print(f"Uploading files from {model_path}")
        api.upload_folder(
            folder_path=model_path,
            repo_id=model_name,
            repo_type="model",
            commit_message="Upload Groot model checkpoint"
        )

        print(f"✅ Model successfully uploaded to https://huggingface.co/{model_name}")

    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        print("\nPlease make sure your Hugging Face token is valid and has write permissions.")

if __name__ == "__main__":
    upload_model_to_hf()
