## 2025-09-27
#### Set UP New Runpod

chmod +x setup_conda.sh
./setup_conda.sh

#### Download from huggingface 
huggingface-cli download \
    --repo-type dataset siyulw2025/so101_test_orange_pick_001 \
    --local-dir ./siyulw2025/so101_test_orange_pick_001

#### Gr00t training
https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning

#### ACT training 
lerobot-train \
  --dataset.repo_id=siyulw2025/so101_test_orange_pick_001 \
  --policy.type=act \
  --output_dir=outputs/train/so101_test_orange_pick_001 \
  --job_name=orange_pick_and_place \
  --policy.device=cuda \
  --policy.repo_id=siyulw2025/so101_test_orange_pick_001 \
  --batch_size=1 \
  --wandb.enable=true \

# Install Claude Code
npm install -g @anthropic-ai/claude-code

-----------------------------------------------------------
/dev/ttyACM0

lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0


https://github.com/huggingface/lerobot/blob/ddba994d73e6315e78c76173cd4fa90d471fc662/src/lerobot/datasets/lerobot_dataset.py#L647

revision='v3.0'

ls -l /dev/ttyACM0
crw-rw-rw- 1 root dialout 166, 0 Sep 24 22:21 /dev/ttyACM0

python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/dataset.hdf5


