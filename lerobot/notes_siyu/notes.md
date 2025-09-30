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
upload then downloads

cd /workspace/siyulw2025/
huggingface-cli download siyulw2025/so101_test_orange_pick_gr00t --local-dir so101_test_orange_pick_gr00t --repo-type dataset

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
  --video-backend torchvision_av \
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
cd /workspace/Isaac-GR00T && python scripts/gr00t_finetune.py \
  --dataset-path ../siyulw2025/so101_test_orange_pick_gr00t/ \
  --num-gpus 1 \
  --output-dir ./so101-checkpoints \
  --max-steps 10000 \
  --data-config custom_data_config:So101DualCamNoGripperDataConfig \
  --video-backend torchvision_av \
  --report_to wandb \
  --batch-size 50 \
  --dataloader-num-workers 1 \
  --dataloader-prefetch-factor 2

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


