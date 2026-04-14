#!/bin/bash
pixi run lerobot-record \
  --robot.type=aic_controller \
  --robot.id=aic \
  --teleop.type=aic_keyboard_ee \
  --teleop.id=aic \
  --robot.teleop_target_mode=cartesian \
  --robot.teleop_frame_id=base_link \
  --dataset.repo_id=siyulw2025/cable-insertion-0407 \
  --dataset.single_task="insert cable into connector" \
  --dataset.push_to_hub=true \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true


  ### remove old cache
  rm -rf ~/.cache/huggingface/lerobot/siyulw2025/ca
  ble-insertion-0407

Gazebo sim evaluation
# Indicate distrobox to use Docker as container manager
export DBX_CONTAINER_MANAGER=docker

# Create and enter the eval container
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
# If you do *not* have an NVIDIA GPU, remove the --nvidia flag for GPU support
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval

# Inside the container, start the environment
/entrypoint.sh ground_truth:=false start_aic_engine:=true


CUDA_VISIBLE_DEVICES=0 pixi run -e pi05 ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunRLT \
  -p policy_args.vla_backend:=pi05 \
  -p policy_args.checkpoint_path:="$(pwd)/aic_utils/aic_rlt/checkpoints/rlt_pi05/phase2_offline.pt" \
  -p policy_args.pi05_checkpoint:=/home/yifeng/workspace/pi05_base/pi05_base \
  -p "policy_args.instruction:=Insert SFP cable into NIC port"
