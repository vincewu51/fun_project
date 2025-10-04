## adding wandb
# Add wandb config to your pi0_orange.yaml
# train:
#   total_steps: 50000
#   batch_size: 32
#   log_interval: 100
#   save_interval: 5000
#   wandb_project: orange_pi0
#   wandb_entity: your_wandb_username
#   wandb_run_name: pi0_orange_run

## fine-tuning
# uv run scripts/train_pytorch.py pi0_orange \
# --exp_name=pi0_orange_run \
# --save_interval 5000

from openpi.training import config as _config
from openpi.policies import policy_config

# Load config + trained checkpoint
config = _config.get_config("pi0_orange")
checkpoint_dir = "checkpoints/pi0_orange/pi0_orange_run/50000"  # pick best step

policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Example observation from robot camera
obs = {
    "observation/exterior_image_1_left": ...,  # numpy image array
    "observation/wrist_image_left": ...,      # numpy image array
    "prompt": "pick up the orange and place it on the plate"
}

action = policy.infer(obs)["actions"]
print("Predicted action:", action)
