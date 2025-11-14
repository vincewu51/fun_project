## 11/14
lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=ywu67/eval_smolvla_grab_pen\
  --dataset.single_task="Put lego brick into the transparent box" \
  --policy.path=ywu67/smolvla_grab_pen
  
## 10/12
● Perfect! Here's the summary of how lerobot motor setup works:

  Key Findings

  Motor Setup Does NOT Save Config Files

  The lerobot-setup-motors command writes settings directly to each motor's EEPROM (permanent hardware memory):
  - Motor IDs
  - Baudrate settings

  No files are created - the motors themselves remember their configuration.

  Calibration DOES Save Config Files

  Location:
  ~/.cache/huggingface/lerobot/calibration/robots/{robot_type}/{robot_id}.json

  Example:
  ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_robot.json

  Format: JSON
  {
    "shoulder_pan": {
      "id": 1,
      "drive_mode": 0,
      "homing_offset": 2047,
      "range_min": 500,
      "range_max": 3500
    }
  }

  Copying Across Machines

  ✅ Motor Setup - NO COPYING NEEDED

  Motors retain their IDs and baudrate in hardware. Just plug them into the new machine!

  ✅ Calibration Files - PORTABLE

  Copy the JSON file:
  # From machine A to machine B
  scp ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/*.json \
      user@machine-b:~/.cache/huggingface/lerobot/calibration/robots/so101_follower/

  Only update the port path when using on new machine:
  lerobot-calibrate \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0  # <- Update this
      --robot.id=my_robot

  The port path is not stored in the calibration file - it's only used at runtime!

## 10/09
<!-- replay recorded files -->
python scripts/environments/teleoperation/replay.py \
    --dataset_file ./datasets/Kitchen-Fridge-Orange-Mimic-v0-1009.hdf5 \
    --task Kitchen-Fridge-Orange-Mimic-v0 \
    --enable_cameras \
    --device cpu

<!-- how to manage multiple tasks/hdf5 files  -->

## 10/08
#### on the dev-xlerobot-debug branch

#### working tasks on debug branch

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Orange-Mimic-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Orange-Mimic-v0-1009.hdf5 --append

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Orange-Teleop-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Orange-Teleop-v0.hdf5

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Cabinet-Plate-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Cabinet-Plate-v0.hdf5

<!-- fixed by https://github.com/brain-sim/leisaac/commit/25eafce91ed4ceab15d28b59a3842a0333101c95 -->



#### not working on debug branch

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Microwave-Plate-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Microwave-Plate-v0.hdf5
<!--Traceback (most recent call last):
  File "/home/yifeng/workspace/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py", line 351, in <module>
    main()
  File "/home/yifeng/workspace/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py", line 169, in main
    env_cfg = parse_env_cfg(
              ^^^^^^^^^^^^^^
  File "/home/yifeng/miniconda3/envs/isaac-brain/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py", line 138, in parse_env_cfg
    cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yifeng/miniconda3/envs/isaac-brain/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py", line 104, in load_cfg_from_registry
    mod = importlib.import_module(mod_name)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yifeng/miniconda3/envs/isaac-brain/lib/python3.11/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1140, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'leisaac.tasks.household.kitchen_microwave_stock_env_cfg'

-->

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Bottle-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Bottle-v0.hdf5

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Bottle-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Bottle-v0.hdf5

<!-- Summary of Changes

  1. Task Registration Fix (leisaac/tasks/household/__init__.py)

  - Lines 44, 53: Changed module reference from non-existent kitchen_subtask_env_cfg to kitchen_counter_bottle_env_cfg
  - Updated class name to FridgeBottlePlacementEnvCfg

  2. Missing Template Export (leisaac/tasks/template/__init__.py)

  - Line 17: Added XLeRobotRewardsCfg to the exports from xlerobot_env_cfg

  3. Outdated Reward API Fix (leisaac/tasks/household/fridge_stocking_env_cfg.py)

  - Lines 118-136: Commented out sequential_progress reward term that was using deprecated parameters (stage_rewards, approach_distance_threshold,
  fridge_distance_threshold)
  - Added comment explaining it needs updating to use new sequence parameter like in kitchen_fridge_stock_env_cfg.py

  4. Missing EE Frame Configuration (leisaac/tasks/household/kitchen_counter_bottle_env_cfg.py)

  - Line 35: Added LEFT_EE_CFG = SceneEntityCfg("left_ee_frame") definition
  - Updated all 4 grasp_object subtasks to include both left_ee_frame_cfg and right_ee_frame_cfg parameters:
    - Orange sequence (line ~190)
    - Bottle fridge sequence (line ~344)
    - Bottle counter sequence (line ~498)
    - Plate counter sequence (line ~624)

  5. Door Subtask Name Corrections (leisaac/tasks/household/kitchen_counter_bottle_env_cfg.py)

  - Renamed door_open → fridge_door_open (2 occurrences)
  - Renamed door_closed → fridge_door_closed (2 occurrences)
  - Changed parameter fridge_cfg → target_cfg in all door subtasks to match registry requirements -->


python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Bottle-Teleop-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Bottle-Teleop-v0.hdf5
<!--ModuleNotFoundError: No module named 'leisaac.tasks.household.kitchen_subtask_env_cfg'-->



#### task list on debug branch

<!-- 
Kitchen-Fridge-Orange-v0
Kitchen-Fridge-Orange-Mimic-v0
Kitchen-Fridge-Orange-Teleop-v0
Kitchen-Fridge-Orange-Test-v0
Kitchen-Fridge-Bottle-v0
Kitchen-Fridge-Bottle-Teleop-v0
Kitchen-Cabinet-Plate-v0
Kitchen-Cabinet-Plate-Teleop-v0
Kitchen-Cabinet-Plate-Mimic-v0
Kitchen-Microwave-Plate-v0
Kitchen-Microwave-Plate-Teleop-v0
Kitchen-Microwave-Plate-Mimic-v0 -->


#### task list on the dev-assets branch

<!-- 
Household-Dishwashing-v0
Household-Microwaving-v0
Household-FridgeStocking-v0
Household-PlateArrangement-v0
Household-ShelfSorting-v0
Household-DishwasherRestock-v0
Household-MicrowaveMealPrep-v0
Household-CoffeeService-v0
Household-BreakfastSetup-v0
Household-FruitDisplay-v0
Household-PantryLoading-v0
Household-UtensilStation-v0 -->




## 2025-10-02
### device
xlerobot_left:'/dev/ttyACM1'
xlerobot_right:'/dev/ttyACM0'

lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1

lerobot-setup-motors \Household
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=xlerobot_left_leader_arm

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=xlerobot_right_leader_arm

### isaac sim collect data
python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task -FruitDisplay-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Household-FruitDisplay-v0.hdf5

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Household-FridgeStocking-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Household-FridgeStocking-v0.hdf5

#### todo: need to check how to change that the wheels cannot turn towards forward, gripping friction is low. in real testing we should use one time use plates
python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Household-PlateArrangement-v0 --enable_cameras --device=cpu 



======================

## 2025-09-30
## hugging face download

## inference Gr00T
python scripts/evaluation/policy_inference.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --eval_rounds=10 \
  --policy_type=gr00tn1.5 \
  --policy_host=localhost \
  --policy_port=5555 \
  --policy_timeout_ms=5000 \
  --policy_action_horizon=16 \
  --policy_language_instruction="Grab the orange and place in the plate" \
  --device=cuda \
  --enable_cameras



python scripts/inference_service.py --model-path /media/yifeng-wu/E/gr00t_orange/checkpoint-3000 --server --embodiment_tag new_embodiment --data-config so100_dualcam 

#====================================================================
### server
## on PC, we install the Isaac-Gr00T to the leisaac environment
# server
python scripts/inference_service.py --model-path /media/yifeng-wu/E/gr00t_orange_cache/checkpoint-30000 --server

# client
python scripts/inference_service.py  --client

### inference
python scripts/inference_service.py --server \
    --model_path /media/yifeng-wu/E/gr00t_orange_cache/checkpoint-30000 \
    --embodiment-tag new_embodiment \
    --data-config custom_data_config:So101DualCamNoGripperDataConfig \
    --denoising-steps 4

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


