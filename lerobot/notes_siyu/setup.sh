!/bin/bash

# Path to your miniconda installation
CONDA_PATH="/workspace/miniconda"

# Add conda to PATH for this session
export PATH="$CONDA_PATH/bin:$PATH"

# Initialize conda for bash if not already done
if [ -f "$CONDA_PATH/bin/conda" ]; then
    $CONDA_PATH/bin/conda init bash
fi

# Persist PATH update in .bashrc (idempotent)
if ! grep -q "$CONDA_PATH/bin" ~/.bashrc; then
    echo "export PATH=\"$CONDA_PATH/bin:\$PATH\"" >> ~/.bashrc
fi

# Optional: set default env and pkg dirs to /workspace
conda config --add envs_dirs "$CONDA_PATH/envs"
conda config --add pkgs_dirs "$CONDA_PATH/pkgs"

# Reload bashrc so conda is available immediately
source ~/.bashrc

echo "Conda setup complete. Run 'conda info' to verify."

# =========================
# GitHub SSH Setup
# =========================
SSH_DIR="/workspace/.ssh"
TEMP_KEY="/root/id_ed25519_temp"
TEMP_KEY_PUB="/root/id_ed25519_temp.pub"

if [ -f "$SSH_DIR/id_ed25519" ]; then
    # Copy persistent key to temp location
    cp "$SSH_DIR/id_ed25519" "$TEMP_KEY"
    cp "$SSH_DIR/id_ed25519.pub" "$TEMP_KEY_PUB"

    # Correct permissions
    chmod 600 "$TEMP_KEY"
    chmod 644 "$TEMP_KEY_PUB"

    # Start ssh-agent if not running
    if ! pgrep -u "$USER" ssh-agent > /dev/null; then
        eval "$(ssh-agent -s)"
    fi

    # Add key
    ssh-add "$TEMP_KEY"
    echo "GitHub SSH key loaded."
fi

# =========================
# HF and WandB Setup
# =========================
# Activate lerobot Conda environment
source /workspace/miniconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda activate lerobot

# Source .env if it exists
if [ -f /workspace/.env ]; then
    source /workspace/.env
    echo "Loaded secrets from /workspace/.env"
    
    # Hugging Face login
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    
    # Weights & Biases login
    wandb login "$WANDB_API_KEY"
else
    echo "⚠️ No /workspace/.env file found. Please create one with your HF_TOKEN and WANDB_API_KEY."
fi