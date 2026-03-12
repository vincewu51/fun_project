## 11/9




########### 10:20 ###########

compare your current smolVLA checkpoint and previous checkpoint. note you changed the features to use the top 32 features at your current checkpoint. your previous checkpoint Had 9 image modalities (rgb, depth, seg_instance_id for all 3 cameras) but now you have 3 no depth or segmentation. also example is task 0000 but current checkpoint is for task 0037. "task_index": 37, "task_name": "clean_a_trumpet", "task": "In the bedroom, pick up the scrub brush from the desk and scrub the cornet (trumpet) on the desk until it's no longer covered in dust."

please create all the files neccesary for inference for task 37.

your current checkpoint at:  /home/yifeng/workspace/b1k-checkpoints-and-results/checkpoints-task0037/smolvla-checkpoint-2500

previous checkpoint at: /home/yifeng/workspace/b1k-checkpoints-and-results/checkpoints-task0000/smolvla-30000/checkpoints/step_030000


previously you have your notes how you fix all the issues:
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

      print(":white_check_mark: Done! Checkpoint files created:")
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