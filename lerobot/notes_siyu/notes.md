11/12

uv run scripts/serve_b1k.py --task_name=picking_up_trash policy:checkpoint --policy.config=pi05_b1k --policy.dir=/home/yifeng/workspace/b1k-checkpoints-and-results/checkpoints-task0001/pi05-step18499



11/8
uv run scripts/serve_b1k.py --task_name=picking_up_trash policy:checkpoint
  --policy.config=pi05_b1k
  --policy.dir=/home/yifeng/workspace/b1k-checkpoints-and-results/checkpoints-task0001/pi05-step4000/checkpoints/step_4000

cd /home/yifeng/workspace/BEHAVIOR-1K/OmniGibson
  python omnigibson/learning/eval.py \
      policy=websocket \
      task.name=picking_up_trash \
      log_path=/home/yifeng/workspace/b1k-checkpoints-and-results/eval-results-task0001/pi05_s4000



## 11/7
export CUDA_VISIBLE_DEVICES=0

python src/lerobot/scripts/serve_smolvla_websocket.py \
      --pretrained_name_or_path=/home/yifeng/workspace/smolvla_training/outputs/train/smolvla_behavior/checkpoint_step_4200 \
      --device=cuda \
      --host=0.0.0.0 \
      --port=8000

python omnigibson/learning/eval.py     policy=websocket     task.name=hiding_Easter_eggs     log_path=/home/yifeng/workspace/b1k-checkpoints-and-results/eval-results-task0006/smolvla_4200

python train_smolvla_lerobot.py     --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006     --filter_state     --batch_size 4     --max_steps 100000     --save_steps 200     --keep_last_n_checkpoints 10     --chunk_size 50     --num_workers 0     --lr 1e-4     --video_tolerance 0.15     --wandb_project "b1k-smolvla-task0006"



SmolVLA Checkpoint Fix for 32-Feature Training

  Problem

  Checkpoint was missing normalization stats files needed for inference, causing "mean is infinity" and dimension mismatch errors.

  Solution Script

  Save this as fix_checkpoint.py in your smolvla_training directory:

  #!/usr/bin/env python3
  """Fix SmolVLA checkpoint to add missing preprocessor with 32-dim filtered stats."""
  import json
  import torch
  from pathlib import Path
  from safetensors.torch import load_file, save_file

  # Import your filter function
  from filter_allowed_state import get_top32_indices

  def fix_checkpoint(dataset_path, checkpoint_path):
      dataset_path = Path(dataset_path).expanduser()
      checkpoint_path = Path(checkpoint_path).expanduser()

      print(f"Fixing checkpoint: {checkpoint_path}")

      # 1. Load dataset stats from JSON and convert to tensors
      with open(dataset_path / "meta/stats.json") as f:
          stats_json = json.load(f)

      stats = {}
      for key, value_dict in stats_json.items():
          if isinstance(value_dict, dict):
              for stat_name, stat_value in value_dict.items():
                  stats[f"{key}.{stat_name}"] = torch.tensor(stat_value, dtype=torch.float32)

      # 2. Filter observation.state stats from 256 dims to 32 dims
      top32_indices = get_top32_indices()
      for key in list(stats.keys()):
          if key.startswith('observation.state.') and key != 'observation.state.count':
              if stats[key].shape[0] == 256:
                  stats[key] = stats[key][top32_indices]  # Filter to 32 dims

      # 3. Fix image stats shape from [3] to [3, 1, 1]
      for key in stats.keys():
          if 'observation.images' in key and ('.mean' in key or '.std' in key):
              if stats[key].shape == torch.Size([3]):
                  stats[key] = stats[key].reshape(3, 1, 1)

      # 4. Save preprocessor config (6 steps, normalizer at step 5)
      preprocessor_config = {
          "name": "policy_preprocessor",
          "steps": [
              {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
              {"registry_name": "to_batch_processor", "config": {}},
              {"registry_name": "smolvla_new_line_processor", "config": {}},
              {"registry_name": "tokenizer_processor", "config": {
                  "max_length": 48, "task_key": "task", "padding_side": "right",
                  "padding": "longest", "truncation": True,
                  "tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
              }},
              {"registry_name": "device_processor", "config": {"device": "cuda", "float_dtype": None}},
              {"registry_name": "normalizer_processor", "config": {
                  "eps": 1e-08,
                  "features": {
                      "observation.images.rgb.left_wrist": {"type": "VISUAL", "shape": [480, 480, 3]},
                      "observation.images.rgb.right_wrist": {"type": "VISUAL", "shape": [480, 480, 3]},
                      "observation.images.rgb.head": {"type": "VISUAL", "shape": [720, 720, 3]},
                      "observation.cam_rel_poses": {"type": "STATE", "shape": [21]},
                      "observation.state": {"type": "STATE", "shape": [32]},  # 32 dims!
                      "observation.task_info": {"type": "STATE", "shape": [382]},
                      "action": {"type": "ACTION", "shape": [23]}
                  },
                  "norm_map": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
              }, "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors"}
          ]
      }

      # 5. Save postprocessor config
      postprocessor_config = {
          "name": "policy_postprocessor",
          "steps": [
              {"registry_name": "unnormalizer_processor", "config": {
                  "features": {"action": {"type": "ACTION", "shape": [23]}},
                  "norm_map": {"ACTION": "MEAN_STD"}
              }}
          ]
      }

      # 6. Write all files
      with open(checkpoint_path / "policy_preprocessor.json", 'w') as f:
          json.dump(preprocessor_config, f, indent=2)

      with open(checkpoint_path / "policy_postprocessor.json", 'w') as f:
          json.dump(postprocessor_config, f, indent=2)

      save_file(stats, str(checkpoint_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
      save_file(stats, str(checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))

      print("✅ Done! Checkpoint files created:")
      print(f"   - policy_preprocessor.json (6 steps)")
      print(f"   - policy_preprocessor_step_5_normalizer_processor.safetensors (32-dim stats)")
      print(f"   - policy_postprocessor.json")
      print(f"   - policy_postprocessor_step_0_unnormalizer_processor.safetensors")

  if __name__ == "__main__":
      import argparse
      parser = argparse.ArgumentParser()
      parser.add_argument("--dataset_path", required=True)
      parser.add_argument("--checkpoint_path", required=True)
      args = parser.parse_args()

      fix_checkpoint(args.dataset_path, args.checkpoint_path)

  Usage

  cd ~/workspace/smolvla_training

  python fix_checkpoint.py \
      --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
      --checkpoint_path ~/workspace/smolvla_training/outputs/train/smolvla_behavior/checkpoint_step_4200

  What it does

  1. Loads dataset stats from stats.json and converts to tensors
  2. Filters observation.state from 256 dims → 32 dims using get_top32_indices()
  3. Fixes image stats shape from [3] to [3, 1, 1]
  4. Creates preprocessor config with 6 steps (normalizer at step 5)
  5. Creates postprocessor config
  6. Saves all 4 required files with correct dimensions

  Key changes

  - observation.state stats: 256 dims → 32 dims (filtered)
  - Preprocessor steps: 3 → 6 steps (added tokenizer, batch processor, etc.)
  - Normalizer position: step 2 → step 5
  - Image stats shape: [3] → [3, 1, 1]


============
## haven't implement yet for smolvla
  ❌ Missing Normalization Stats: ROOT CAUSE

  The actual problem is that your checkpoint is missing the preprocessor files. Looking at your training script:

  Line 192 (train_smolvla_lerobot.py:192):
  preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

  The preprocessor is created with stats, but...

  Line 369 (train_smolvla_lerobot.py:369):
  policy.save_pretrained(checkpoint_path)

  Only the policy is saved, NOT the preprocessor! The training script doesn't save the preprocessor/postprocessor.

  Compare this to the official LeRobot training code (lerobot/scripts/lerobot_train.py:385-394):
  save_checkpoint(
      checkpoint_dir=checkpoint_dir,
      step=step,
      cfg=cfg,
      policy=accelerator.unwrap_model(policy),
      optimizer=optimizer,
      scheduler=lr_scheduler,
      preprocessor=preprocessor,  # ← This saves the preprocessor!
      postprocessor=postprocessor,
  )

  ---
  Solutions

  Solution 1: Modify Training Script to Save Preprocessor (For Future Training)

  Add preprocessor saving to your training script:

  # In train_smolvla_lerobot.py, around line 369
  if args.save_steps is not None and step > 0 and step % args.save_steps == 0:
      checkpoint_path = output_dir / f"checkpoint_step_{step}"
      policy.save_pretrained(checkpoint_path)
      preprocessor.save_pretrained(checkpoint_path)  # ADD THIS LINE
      postprocessor.save_pretrained(checkpoint_path)  # ADD THIS LINE
      print(f"  ✓ Saved checkpoint to {checkpoint_path}")


## 11/6

uv run scripts/serve_b1k.py     --task_name=turning_on_radio     policy:checkpoint     --policy.config=pi0_b1k     --policy.dir=/home/yifeng/workspace/BEHAVIOR-1K/checkpoints/openpi-radio-task0/checkpoint-20000



python OmniGibson/omnigibson/learning/eval.py     policy=websocket     task.name=turning_on_radio     log_path=/home/yifeng/workspace/logs/openpi0_20k_radio     headless=true     write_video=True     partial_scene_load=true


Terminal 1 (Policy Server):
  source .venv/bin/activate
  uv run scripts/serve_b1k.py --task_name=picking_up_trash policy:checkpoint --policy.config=pi0_b1k --policy.dir=/home/yifeng/Downloads/openpi_picking_up_trash/49999

  Terminal 2 (OmniGibson Evaluation):
  conda activate behavior
  export CUDA_VISIBLE_DEVICES=0
  python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=picking_up_trash log_path=/home/yifeng/workspace/log_pi0_baseline_pickuptrash


## 11/3
python src/lerobot/scripts/serve_smolvla_websocket.py \
      --pretrained_name_or_path=/home/yifeng/workspace/smolvla-task0000/checkpoints/step_030000 \
      --device=cuda \
      --host=0.0.0.0 \
      --port=8000


python omnigibson/learning/eval.py     policy=websocket     task.name=turning_on_radio     log_path=~/eval_results

pip install msgpack==1.1.2 websockets==15.0.1 opencv-python==4.11.0.86


## 11/2
#### policy server
conda activate behavior 
cd ~/workspace/b1k-baselines
export CUDA_VISIBLE_DEVICES=0
python baselines/il_lib/serve.py \
  robot=r1pro \
  task=behavior \
  arch=wbvima \
  ckpt_path=/home/yifeng/Downloads/wbvima_radio.pth


#### eval
conda activate behavior 
python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=turning_on_radio env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper log_path=~/workspace/wbvima_log/


-------------

#### fast dev checking

cd ~/workspace/b1k-baselines/baselines/il_lib

python train.py \
  data_dir=/home/yifeng/local_behavior_data/pcd_vid \
  robot=r1pro \
  task=behavior \
  task.name=turning_on_radio \
  arch=wbvima \
  trainer.fast_dev_run=true \
  +eval=behavior \
  headless=false \
  ckpt_path=/home/yifeng/Downloads/wbvima_radio/archive


<!-- python train.py \
  data_dir=~/radio_data \
  robot=r1pro \
  task=behavior \
  task.name=turning_on_radio \
  arch=wbvima \
  trainer.fast_dev_run=true \
  +eval=behavior \
  headless=false \
  ckpt_path=/home/yifeng/Downloads/wbvima_radio/archive -->


###### training
python train.py \
  data_dir=~/radio_data \
  robot=r1pro \
  task=behavior \
  task.name=turning_on_radio \
  arch=wbvima \
  trainer.fast_dev_run=true \
  +eval=behavior \
  headless=false \
  ckpt_path=/home/yifeng/Downloads/wbvima_radio/archive



<!-- python train.py data_dir=~/radio_data robot=r1pro task=behavior task.name=turning_on_radio     arch=wbvima trainer.fast_dev_run=true +eval=behavior headless=false     ckpt_path=/home/yifeng/Downloads/wbvima_radio/archive -->



Summary of Changes

  Problem:

  Your original data path had special characters that Hydra couldn't parse:
  /run/user/1000/gvfs/smb-share:server=tnas-039c-3.local,share=public/2025-challenge-demos/data/task-0000

  Solution: Created a symlink to simplify the path

  Step 1: Created symlink ~/radio_data
  ln -sf "/run/user/1000/gvfs/smb-share:server=tnas-039c-3.local,share=public/" ~/radio_data

  Key insight: The OmniGibson dataset loader (in iterable_dataset.py:61) automatically appends
  /2025-challenge-demos/data/task-0000 to your data_dir, so the symlink must point to the parent 
  directory that contains 2025-challenge-demos/, not the task folder itself.

  Result:

  - Before: data_dir='/run/user/1000/gvfs/smb-share:server=tnas-039c-3.local,share=public/2025-challenge-
  demos/data/task-0000' ❌ (Hydra parsing error)
  - After: data_dir=~/radio_data ✅ (Clean path that Hydra can parse)

  The final resolved path is:
  ~/radio_data/2025-challenge-demos/data/task-0000/ 
  → Contains your 200+ episode_*.parquet files

===================

## 10/31

  Name: omnigibson
  Version: 3.7.1

  Name: isaacsim
  Version: 4.5.0.0

  
Issues identified:
1. # Overwrite below to modify the robot controller 
controllers:
https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/learning/configs/robot/r1pro.yaml

2.


<!-- about GPU concern - set up vironment Variables to Control Memory -->
  export CUDA_VISIBLE_DEVICES=1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.6

  uv run scripts/serve_b1k.py --task_name=turning_on_radio policy:checkpoint --policy.config=pi0_b1k --policy.dir=/home/yifeng/Downloads/openpi_turning_on_radio/49999_radio


  export CUDA_VISIBLE_DEVICES=0
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.6
  python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    log_path=/home/yifeng/workspace/log \
    headless=true \
    write_video=false \
    partial_scene_load=true

  Key settings:
  - garbage_collection_threshold:0.6 - Release cached memory more aggressively when
   60% is used
  - max_split_size_mb:512 - Prevent large fragmentation



<!-- Key PyTorch memory settings:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 - Reduces fragmentation by
  limiting split sizes
  - expandable_segments:True - Allows memory to grow more efficiently -->
# For the policy server (serve_b1k.py)
  source .venv/bin/activate
  export CUDA_VISIBLE_DEVICES=0  # Use same GPU
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
  export PYTORCH_NO_CUDA_MEMORY_CACHING=0
  uv run scripts/serve_b1k.py --task_name=turning_on_radio policy:checkpoint
  --policy.config=pi0_b1k
  --policy.dir=/home/yifeng/Downloads/openpi_turning_on_radio/49999_radio

  # For the evaluation (eval.py)
  conda activate behavior
  export CUDA_VISIBLE_DEVICES=0  # Use same GPU
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
  <!-- python OmniGibson/omnigibson/learning/eval.py policy=websocket
  task.name=turning_on_radio log_path=/home/yifeng/workspace/log -->

python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    log_path=/home/yifeng/workspace/log \
    headless=true \
    write_video=false \
    partial_scene_load=true \



------------

## 10/30

cd ~/workspace/b1k-baselines/baselines/openpi
source .venv/bin/activate


uv run scripts/compute_norm_stats.py --config-name pi0_b1k

<!-- export CUDA_VISIBLE_DEVICES=1 -->
source .venv/bin/activate
uv run scripts/serve_b1k.py --task_name=turning_on_radio policy:checkpoint --policy.config=pi0_b1k --policy.dir=/home/yifeng/Downloads/openpi_turning_on_radio/49999_radio

conda activate behavior
export CUDA_VISIBLE_DEVICES=0 
python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=turning_on_radio log_path=/home/yifeng/workspace/log_pi0_baseline



-------------------------------------
## 10/25

  python scripts/gr00t_finetune.py \
    --dataset-path /home/yifeng/workspace/xlerobot\
    --num-gpus 1 \
    --output-dir ./xlerobot-checkpoints \
    --max-steps 3000 \
    --save-steps 1000 \
    --data-config xlerobot \
    --video-backend torchvision_av \
    --report_to wandb

LORA:
(gr00t) yifeng@Purrticle2:~/workspace/Isaac-GR00T$   python scripts/gr00t_finetune.py \
    --dataset_path /home/yifeng/workspace/xlerobot \
    --data_config xlerobot \
    --video_backend torchvision_av \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --dataloader_num_workers 2 \
    --output_dir ./checkpoints/xlerobot_lora \
    --max_steps 10000 \
    --save_steps 1000


  python scripts/gr00t_finetune.py \
    --dataset_path /workspace/xlerobot \
    --data_config xlerobot \
    --video_backend torchvision_av \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 1 \
    --dataloader_prefetch_factor 1 \
    --output_dir ./xlerobot-checkpoints \
    --max_steps 10000 \
    --save_steps 500



## 10/23
Summary: Converting v3.0 → v2.0 and Uploading to HuggingFace

  Files to use for future conversions:

  1. Convert v3.0 → v2.1:
  python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
      --repo-id=YOUR_REPO_ID \
      --root=/tmp/lerobot_datasets
  2. Convert v2.1 → v2.0:
  python convert_local_v21_to_v20.py
  2. (Edit the paths inside the script to point to your v2.1 dataset)

  2. Or use the standalone script:
  python src/lerobot/datasets/v30/convert_dataset_v21_to_v20.py \
      --repo-id=YOUR_REPO_ID \
      --output-dir=/tmp/lerobot_v20_datasets \
      --delete-episodes-stats
  3. Upload to HuggingFace:
  python upload_with_hub_api.py
  3. (Edit repo_id and paths inside the script for v2.1 and v2.0 datasets)

  Key Files Created:

  - /Users/yifengwu/workspace/lerobot/src/lerobot/datasets/v30/convert_dataset_v21_t
  o_v20.py - v2.1→v2.0 converter (with bug fixes)
  - /Users/yifengwu/workspace/lerobot/convert_local_v21_to_v20.py - Local conversion
   helper
  - /Users/yifengwu/workspace/lerobot/upload_with_hub_api.py - HuggingFace uploader 
  (working!)

  Quick Todo for Next Time:

  1. Run v3.0→v2.1 conversion
  2. Run v2.1→v2.0 conversion
  3. Edit upload_with_hub_api.py with correct paths
  4. Run upload script (ensure hf auth login with write token)
  5. Done! ✅

● How is Claude doing this session? (optional)
  1: Bad    2: Fine   3: Good   0: Dismiss

## 10/21
# Delete episodes 0, 2, and 5 (modifies original dataset)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"

# Delete episodes and save to a new dataset (preserves original dataset)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --new_repo_id lerobot/pusht_after_deletion \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"

## 10/19

### follower
"left_arm" = "/dev/tty.usbmodem5A7C1184421"
"right_arm "= "/dev/tty.usbmodem5A7C1235321"
"base" = "/dev/tty.usbmodem58FA0957301"

left_camera: Camera #0
top_camera: Camera #1
right_camera: Camera #2

### leader
"left_arm" = "/dev/tty.usbmodem5AAF2703161"
"right_arm" = "/dev/tty.usbmodem5AAF2703681"

"top_camera" = "/dev/tty.usbmodem58FA0953841"





--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ 0
  Type: OpenCV
  Id: 0
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Width: 1280
    Height: 720
    Fps: 30.00003
--------------------
Camera #1:
  Name: OpenCV Camera @ 1
  Type: OpenCV
  Id: 1
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Width: 1920
    Height: 1080
    Fps: 5.0
--------------------
Camera #2:
  Name: OpenCV Camera @ 2
  Type: OpenCV
  Id: 2
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Width: 1280
    Height: 720
    Fps: 30.00003
--------------------
Camera #3:
  Name: OpenCV Camera @ 3
  Type: OpenCV
  Id: 3
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Width: 1920
    Height: 1080
    Fps: 15.0
--------------------


<!-- =========================== -->
## 10/18

### follower
left_arm = '/dev/tty.usbmodem5A7C1184421'
right_arm = '/dev/tty.usbmodem5A7C1235321'
base = '/dev/tty.usbmodem58FA0957301'

### leader
left_arm = '/dev/tty.usbmodem5AAF2703161'
right_arm = '/dev/tty.usbmodem5AAF2703681'


<!-- https://bambot.org/feetech.js -->



## 10/16
  python scripts/gr00t_finetune.py \
    --dataset-path /home/yifeng/workspace/xlerobot_gr00t_data \
    --num-gpus 1 \
    --output-dir ./xlerobot-checkpoints \
    --max-steps 10000 \
    --data-config xlerobot \
    --video-backend decord \
    --report_to wandb

## 10/13

#### replay
python scripts/environments/teleoperation/replay.py \
    --dataset_file ./datasets/Kitchen-Fridge-Orange-Teleop-v0-5-both_CheckLastOne.hdf5 \
    --task Kitchen-Fridge-Orange-Mimic-v0 \
    --enable_cameras \
    --device cpu

<!-- view the last two 
  If you know the total number of episodes (let's say there are 10 episodes, so indices 0-9): -->

python -c "from isaaclab.utils.datasets import HDF5DatasetFileHandler; h = HDF5DatasetFileHandler(); h.open('./datasets/Kitchen-Fridge-Orange-Teleop-v0-5-both_CheckLastOne.hdf5'); num = 
  h.get_num_episodes(); print(f'Total episodes: {num}'); print(f'Last 2 episodes indices: {num-2} {num-1}')"


  python scripts/environments/teleoperation/replay.py \
      --dataset_file ./datasets/Kitchen-Fridge-Orange-Teleop-v0-5-both_CheckLastOne.hdf5 \
      --task Kitchen-Fridge-Orange-Mimic-v0 \
      --enable_cameras \
      --device cpu \
      --select_episodes 14 15

<!-- shouldn't just directly concat the hdf5
will have indexing issues
solution:
https://github.com/search?q=repo%3Ahuggingface%2Flerobot%20MultiLeRobotDataset&type=code
https://github.com/huggingface/lerobot/issues/847 -->


## 10/12
### inference using local file
python scripts/inference_service.py --server \
    --model_path /Users/siyulw/Documents/gr00t_orange/checkpoint-3000 \
    --embodiment-tag new_embodiment \
    --data-config so100_dualcam \
    --denoising-steps 4
    
python getting_started/examples/eval_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0953841 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="pick up the orange and place it into the plate."

## 10/11
<!-- collected 30+ pick and place orange 
prompt: "use right arm pick up an orange and gently place it into a plate." -->

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Orange-Teleop-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Orange-Teleop-v0-5-both.hdf5

<!-- working to collect plate organization data -->
python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Cabinet-Plate-Teleop-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Cabinet-Plate-Teleop-v0-1.hdf5

<!-- Traceback (most recent call last):
  File "/home/yifeng/workspace/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py", line 355, in <module>
    main()
  File "/home/yifeng/workspace/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py", line 344, in main
    env.step(actions)
  File "/home/yifeng/miniconda3/envs/isaac-brain/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py", line 174, in step
    self.action_manager.process_action(action.to(self.device))
  File "/home/yifeng/miniconda3/envs/isaac-brain/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/managers/action_manager.py", line 329, in process_action
    raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
ValueError: Invalid action shape, expected: 18, received: 17. -->

#### delete the last one
import h5py

file_path = "./datasets/Kitchen-Fridge-Orange-Teleop-v0-3-both.hdf5"

with h5py.File(file_path, "r+") as f:
    demos = list(f.keys())
    print("All demos:", demos)
    # Usually named like 'demo_0', 'demo_1', ...
    last_demo = demos[-1]
    print(f"Deleting {last_demo}")
    del f[last_demo]
    print("Deleted successfully!")

## 10/10

<!-- collected 10+ pick and place orange -->
prompt: "use right arm pick up an orange and gently place it into a plate."


python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Orange-Teleop-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Orange-Teleop-v0-3-both.hdf5


<!-- Available Tasks (workspace/leisaac/source/leisaac/leisaac/tasks/household/__init__.py)
  Kitchen Tasks:
  - Kitchen-Fridge-Orange-v0
  - Kitchen-Fridge-Orange-Mimic-v0
  - Kitchen-Fridge-Orange-Teleop-v0
  - Kitchen-Fridge-Orange-Test-v0 
  - Kitchen-Fridge-Bottle-Teleop-v0
  - Kitchen-Fridge-Bottle-Plate-v0
  - Kitchen-Oven-Plate-Teleop-v0
  - Kitchen-Dishwasher-Plate-Teleop-v0
  - Kitchen-Microwave-Plate-Teleop-v0
  - Kitchen-Cabinet-Plate-Teleop-v0

  Other Household Tasks:
  - Laundry-Washer-Cloth-Teleop-v0
  - Bathroom-Flush-Teleop-v0 -->

## 10/9
WIP
        modified:   source/leisaac/leisaac/assets/robots/xlerobot.py
        modified:   source/leisaac/leisaac/utils/env_utils.py
Saved working directory and index state WIP on dev-assets-modification: 25eafce correct task registration class names for kitchen tasks

## 10/08
#### on the dev-xlerobot-debug branch

#### working tasks on debug branch

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Orange-Mimic-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Orange-Mimic-v0.hdf5

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Fridge-Orange-Teleop-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Fridge-Orange-Teleop-v0.hdf5

python scripts/environments/teleoperation/teleop_se3_agent.py --num_envs 1 --teleop_device xlerobot --left_arm_port /dev/ttyACM1 --right_arm_port /dev/ttyACM0 --task Kitchen-Cabinet-Plate-v0 --enable_cameras --device=cpu --record --dataset_file=./datasets/Kitchen-Cabinet-Plate-v0.hdf5

<!-- fixed by https://github.com/brain-sim/leisaac/commit/25eafce91ed4ceab15d28b59a3842a0333101c95 -->



#### not working on debug branch
#### TODO:  microwave file (microwave_meal_prep_env_cfg.py) uses BiArmTaskEnvCfg (dual SO-101 arms), which is incompatible with xlerobot
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

## 10-05
## donwload usd files as large files
#### Install Git LFS if not already installed
sudo apt-get update
sudo apt-get install git-lfs -y
git lfs install

#### Clone only the dev branch
git clone --branch dev --single-branch https://github.com/LightwheelAI/leisaac.git
cd leisaac

#### Fetch LFS objects for the dev branch
git lfs pull origin dev

#### updated usd
cd leisaac
git fetch origin
git checkout dev
git pull origin dev
git lfs pull


## 09-28

# train smolvla
lerobot-train \
  --dataset.repo_id=siyulw2025/so101_test_orange_pick_001 \
  --policy.type=smolvla \
  --output_dir=outputs/train/so101_test_orange_pick_002 \
  --job_name=orange_pick_and_place \
  --policy.device=cuda \
  --policy.repo_id=siyulw2025/so101_test_orange_pick_002 \
  --batch_size=50 \
  --wandb.enable=true


## revert the data conversion to v2.1 to have episodes.json
this has no async but is compatible data format with act and smolvla
### upload then downloads

cd /workspace/siyulw2025/f
huggingface-cli download siyulw2025/so101_test_orange_pick_gr00t --local-dir so101_test_orange_pick_gr00t --repo-type dataset

##### download model
huggingface-cli download \
  siyulw2025/gr00t_orange \
  --repo-type model \
  --local-dir /media/yifeng-wu/E//gr00t_orange \
  --exclude "checkpoint-1000/*" "checkpoint-2000/*"

<!-- huggingface-cli download \
    --repo-type model siyulw2025/gr00t_orange \
    --local-dir ~/gr00t_orange -->

## finetune Gr00T
<!-- python scripts/gr00t_finetune.py \
   --dataset-path ../siyulw2025/so101_test_orange_pick_001/ \
   --num-gpus 1 \
   --output-dir ./so101-checkpoints  \
   --max-steps 10000 \
   --data-config so100_dualcam \
   --video-backend torchvision_av -->


python scripts/gr00t_finetune.py \
  --dataset-path ../siyulw2025/so101_test_orange_pick_gr00t/ \
  --num-gpus 1 \
  --output-dir ./so101-checkpoints \
  --max-steps 10000 \
  --data-config so100_dualcam \
  --video-backend discord \
  --report_to wandb


nohup python scripts/gr00t_finetune.py \
  --dataset-path ../siyulw2025/so101_test_orange_pick_gr00t/ \
  --num-gpus 1 \
  --output-dir ./so101-checkpoints \
  --max-steps 10000 \
  --data-config so100_dualcam \
  --video-backend torchvision_av \
  --report_to wandb \
  > train.log 2>&1 &

### create customized data-config as my data doesn't have gripper state 
#### - AssertionError: gripper config not found in state
cd /workspace/Isaac-GR00T && python scripts/gr00t_finetune.py \
    --dataset-path ../siyulw2025/so101_test_orange_pick_gr00t/ \
    --num-gpus 1 \
    --output-dir ./so101-checkpoints \
    --max-steps 10000 \
    --data-config custom_data_config:So101DualCamNoGripperDataConfig \
    --video-backend torchvision_av \
    --report_to wandb

#### reduced size to avoid memory issue (on a small machine for testing)
nohup bash -c 'cd /workspace/Isaac-GR00T && python scripts/gr00t_finetune.py \
  --dataset-path ../siyulw2025/so101_test_orange_pick_gr00t/ \
  --num-gpus 1 \
  --output-dir ./so101-decord-checkpoints \
  --max-steps 10000 \
  --save-steps 1000 \
  --data-config so100_dualcam \
  --video-backend decord \
  --report_to wandb \
  --batch-size 128 \
  --dataloader-num-workers 1 \
  --dataloader-prefetch-factor 2' > train.log 2>&1 &



cd /workspace/Isaac-GR00T && python scripts/gr00t_finetune.py \
  --dataset-path ../siyulw2025/so101_test_orange_pick_gr00t/ \
  --num-gpus 1 \
  --output-dir ./so101-decord-checkpoints \
  --max-steps 100000 \
  --save-steps 10000 \
  --data-config so100_dualcam \
  --video-backend decord \
  --report_to wandb \
  --batch-size 50 \
  --dataloader-num-workers 1 \
  --dataloader-prefetch-factor 2 \
  --resume

## modality file 09301010am (need to update)
{
    "state": {
        "single_arm": {
            "start": 0,
            "end": 6,
            "original_key": "observation.state"
        }
    },
    "action": {
        "single_arm": {
            "start": 0,
            "end": 6,
            "original_key": "action"
        }
    },
    "video": {
        "front": {
            "original_key": "observation.images.front"
        },
        "wrist": {
            "original_key": "observation.images.wrist"
        }
    },
    "annotation": {
        "human.action.task_description": {
            "original_key": "annotation.human.action.task_description"
        },
        "human.validity": {},
        "human.coarse_action": {
            "original_key": "annotation.human.action.task_description"
        }
    }
}

## modality file 09300800am (need to update)
{
    "state": {
        "left_arm": {
            "start": 0,
            "end": 7
        },
        "left_hand": {
            "start": 7,
            "end": 13
        },
        "left_leg": {
            "start": 13,
            "end": 19
        },
        "neck": {
            "start": 19,
            "end": 22
        },
        "right_arm": {
            "start": 22,
            "end": 29
        },
        "right_hand": {
            "start": 29,
            "end": 35
        },
        "right_leg": {
            "start": 35,
            "end": 41
        },
        "waist": {
            "start": 41,
            "end": 44
        }
    },
    "action": {
        "left_arm": {
            "start": 0,
            "end": 7
        },
        "left_hand": {
            "start": 7,
            "end": 13
        },
        "left_leg": {
            "start": 13,
            "end": 19
        },
        "neck": {
            "start": 19,
            "end": 22
        },
        "right_arm": {
            "start": 22,
            "end": 29
        },
        "right_hand": {
            "start": 29,
            "end": 35
        },
        "right_leg": {
            "start": 35,
            "end": 41
        },
        "waist": {
            "start": 41,
            "end": 44
        }
    },
    "video": {
        "front": {
            "original_key": "observation.images.front"
        },
        "wrist": {
            "original_key": "observation.images.wrist"
        }
    },
    "annotation": {
        "human.action.task_description": {},
        "human.validity": {},
        "human.coarse_action": {
            "original_key": "annotation.human.action.task_description"
        }
    }
}


## wandb can't handle loss tensor from the gr00t
import wandb
# Example: convert tensor to float
wandb.log({
    "losses_after_forward": losses_after_forward.item(),
    "losses_after_rm_padding": losses_after_rm_padding.item()
})


## 2025-09-27
### Inference Infra set up from Benchmarking Vision, Language, & Action Models in Procedurally Generated, Open Ended Action Environments (https://arxiv.org/html/2505.05540v1)

For systematic performance evaluation and model profiling, we utilized dedicated hardware resources optimized for each model’s computational requirements:

• OpenVLA inference was executed on an NVIDIA L4 GPU instance, selected for its balance of computational efficiency and memory capacity, ideal for models of intermediate complexity.
• Pi0 Base utilized a single NVIDIA A100 GPU instance equipped with 40 GB of memory, providing ample computational power and memory bandwidth for accurate and efficient inference.
• Pi0 Fast, due to its greater computational demands and inference complexity, was allocated four NVIDIA A100 GPU instances, each with 40 GB of memory, facilitating parallel processing to substantially enhance inference throughput.
• In contrast, GPT 4x inference was managed externally through batch job APIs, eliminating local processing overhead and leveraging scalable cloud resources to efficiently handle its computational requirements.


### github connection

ssh-keygen -t ed25519 -C "wuyifeng5@gmail.com" -f /workspace/.ssh/id_ed25519 -N ""



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

#### Can use Mac to train
device.type = 'mps'

lerobot-train
--dataset.repo_id=siyulw2025/so101_test_orange_pick_001
--policy.type=act
--output_dir=outputs/train/so101_test_orange_pick_001
--job_name=orange_pick_and_place
--policy.device=cuda
--policy.repo_id=siyulw2025/so101_test_orange_pick_001
--batch_size=1
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


