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
