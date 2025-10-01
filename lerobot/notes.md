## 2025-09-30

### server
## on PC, we install the Isaac-Gr00T to the leisaac environment
# server
python scripts/inference_service.py --model-path /media/yifeng-wu/E/gr00t_orange_cache/checkpoint-30000 --server

# client
python scripts/inference_service.py  --client

<!-- python scripts/inference_service.py --server \
    --model_path /media/yifeng-wu/E/gr00t_orange_cache/checkpoint-30000 \
    --embodiment-tag new_embodiment \
    --data-config so100_dualcam \
    --denoising-steps 4 -->

 python scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --policy_type=lerobot-smolvla \
      --policy_host=localhost \
      --policy_port=8080 \
      --policy_timeout_ms=5000 \
      --policy_checkpoint_path=/home/yifeng-wu/smolVLA_orange \
      --policy_action_horizon=10 \
      --policy_language_instruction="orange_pick_and_place" \
      --device=cuda \
      --enable_cameras \
      --video-backend decord

## slack note
<!-- python scripts/gr00t_finetune.py \
   --dataset-path ./demo_data/so101-table-cleanup/ \
   --num-gpus 1 \
   --output-dir ./so101-checkpoints  \
   --max-steps 10000 \
   --data-config so100_dualcam \
   --video-backend torchvision_av -->

## 2025-09-29
#### Download Huggingface model
huggingface-cli download \
    --repo-type model siyulw2025/smolVLA_orange \
    --local-dir ~/smolVLA_orange

  python scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --policy_type=lerobot-smolvla \
      --policy_host=localhost \
      --policy_port=8080 \
      --policy_timeout_ms=5000 \
      --policy_checkpoint_path=/home/yifeng-wu/smolVLA_orange \
      --policy_action_horizon=1000 \
      --policy_language_instruction="orange_pick_and_place" \
      --device=cuda \
      --enable_cameras
## 2025-09-28
#### Async Server
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080

python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.id=simulation \
    --robot.cameras="{ laptop: {type: opencv, index_or_path: 0, width: 1920, height: 1080, 
fps: 30}, phone: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --task="pick and place oranges" \
    --policy_type=act\
    --pretrained_name_or_path=siyulw2025/ACT_orange \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True



lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm 
#### Inference

#### 0928
  python scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --policy_type=lerobot-act \
      --policy_host=localhost \
      --policy_port=8080 \
      --policy_timeout_ms=5000 \
      --policy_checkpoint_path=/home/yifeng-wu/ACT_orange \
      --policy_action_horizon=10 \
      --policy_language_instruction="orange_pick_and_place" \
      --device=cuda \
      --enable_cameras

  python scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --policy_type=lerobot-act \
      --policy_host=localhost \
      --policy_port=8080 \
      --policy_timeout_ms=5000 \
      --policy_checkpoint_path=/home/yifeng-wu/trash_picking \
      --policy_action_horizon=5 \
      --policy_language_instruction="orange_pick_and_place" \
      --device=cuda \
      --enable_cameras

PHYSX_GPU_FOUND=1 PHYSX_USE_GPU=1 CUDA_VISIBLE_DEVICES=0 GPU_FORCE_64BIT_PTR=1 CUDA_LAUNCH_BLOCKING=0 ISAAC_SIM_REALTIME_RATIO=0.1 PHYSX_GPU_HEAP_SIZE=64 python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --policy_type=lerobot-act --policy_host=localhost --policy_port=8080 --policy_timeout_ms=5000 --policy_language_instruction='Pick the orange to the plate' --policy_checkpoint_path=/home/yifeng-wu/ACT_orange --policy_action_horizon=10 --device=cuda --enable_cameras


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
  --batch_size=5 \
  --wandb.enable=true

#### Install Claude Code
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 22

# Verify the Node.js version:
node -v # Should print "v22.20.0".

# Verify npm version:
npm -v # Should print "10.9.3".

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


