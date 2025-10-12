## 10/11
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


