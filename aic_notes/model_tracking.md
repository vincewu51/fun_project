# AIC X-VLA Model Tracking

## Version Overview

| Version | HF Repo | Enc | Data | Method | Iters | Train Time | Chunk | Replan | Val Pos | Score | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v1** | `v1-lora-full-delta` | delta | 270K | LoRA, lr=5e-4 | 40K | ~45 min | 30 | 15 | - | 63/300 | 04-26 |
| **v2** | `v2-lora-full-abs` | abs | 270K | LoRA, lr=5e-4 | 20K | ~40 min | 30 | 15 | - | 63/300 | 04-27 |
| **v3** | `v3-lora-sub-abs` | abs | 103K | LoRA, lr=5e-4, coef=0.1 | 50K | ~1.7h | 30 | 15 | 1.6cm | - | 04-29 |
| **v4** | local only | abs | 270K | Full FT + frozen VLM | 20K | ~30 min | 30 | 15 | - | - | 04-29 |

**Defaults:** action chunk=30 steps (1.5s), replan=15, diffusion steps=10, control=pose

| Version | HF Repo | Encoding | Data | Method | Best Iters | Val Pos Loss | Eval Score | Date |
|---|---|---|---|---|---|---|---|---|
| **v1** | `v1-lora-full-delta` | delta | 270K (full) | LoRA, lr=5e-4, coef=1.0 | 40K | — | 63/300 | 2026-04-26 |
| **v2** | `v2-lora-full-abs` | absolute | 270K (full) | LoRA, lr=5e-4, coef=1.0 | 20K | — | 63/300 | 2026-04-27 |
| **v3** | `v3-lora-sub-abs` | absolute | 103K (subsampled) | LoRA, lr=5e-4, coef=0.1 | 50K | 1.6cm | — | 2026-04-29 |
| **v4** | local only | absolute | 270K (full) | Full FT + frozen VLM | 20K | — | — | 2026-04-29 |

## Detailed Results

### v1 — `v1-lora-full-delta`
- **Action encoding**: delta (relative)
- **Training**: LoRA rank-8, 5e-4 LR, learning_coef=1.0, 40K iters
- **Data**: Full 180 episodes, 270K frames
- **Eval**: Scored 63/300 (tier 1 pass, no insertion)
- **Notes**: First run. Does move but doesn't insert well. This is the baseline.

### v2 — `v2-lora-full-abs`
- **Action encoding**: absolute
- **Training**: LoRA rank-8, 5e-4 LR, learning_coef=1.0, 20K iters
- **Data**: Full 180 episodes, 270K frames
- **Eval**: Scored 63/300
- **Notes**: Same score as v1. Absolute encoding doesn't help alone.

### v3 — `v3-lora-sub-abs`
- **Action encoding**: absolute
- **Training**: LoRA rank-8, 5e-4 LR, learning_coef=0.1, 50K iters
- **Data**: Subsampled 103K frames (P3 stationary tail trimmed)
- **Eval**: Not tested (robot didn't move in initial check)
- **Notes**: Low VLM LR (coef=0.1) prevented backbone adaptation. Robot stuck.

### v4 — local only
- **Action encoding**: absolute
- **Training**: Full FT + frozen VLM, 1e-4 LR, 20K iters (stopped early)
- **Data**: Full 270K frames
- **Eval**: Not tested
- **Notes**: Frozen VLM, trainer overrode freeze causing VLM to drift.

## Eval Scoring Rubric (from competition docs)

| Tier | Category | Max Score |
|---|---|---|
| 1 | Model validity | 1 |
| 2 | Trajectory smoothness | 5 |
| 2 | Task duration | 10 |
| 2 | Trajectory efficiency | 5 |
| 2 | Insertion force | -12 to 0 |
| 2 | Off-limit contacts | -24 to 0 |
| 3 | Cable insertion | -10 to 60 |
| | **Total** | **~300** |

## Control Mode

The ROS policy (`RunXVLA`) supports two control modes via `AIC_XVLA_CMD_MODE` env var:

| Mode | Mechanism | Default | Notes |
|---|---|---|---|
| **pose** | Sends absolute `Pose` target via `set_pose_target` | ✅ Yes | Validated for 63/300 baseline. Smooth. |
| **velocity** | Derives Cartesian `Twist` from `(target - current) / dt` | No | Stiffer gains, more aggressive motion. May help with small-action models. |

Set via: `export AIC_XVLA_CMD_MODE=velocity`

## How to Evaluate

```bash
# Terminal 1 — Serve model
python -m aic_xvla.serve --checkpoint siyulw2025/<REPO> --single-model

# Terminal 2 — Policy (pose mode)
export PYTHONPATH=$PWD/aic_utils/aic_xvla
export AIC_XVLA_SERVER_URL=http://127.0.0.1:8010
export AIC_XVLA_CMD_MODE=pose
export AIC_XVLA_REPLAN=15
export AIC_XVLA_TASK_TIMEOUT_S=180
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_xvla.ros.RunXVLA

# Terminal 3 — Eval engine
docker run --rm --gpus all --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro ghcr.io/intrinsic-dev/aic/aic_eval:latest /entrypoint.sh ground_truth:=false start_aic_engine:=true
```

## Uploading to Hugging Face

```bash
python -c "
from huggingface_hub import HfApi, create_repo, upload_folder
api = HfApi()
repo = 'siyulw2025/<NEW-REPO-NAME>'
create_repo(repo, repo_type='model', exist_ok=True)
upload_folder(folder_path='<PATH>/ckpt-NNNNN', path_in_repo='.', repo_id=repo, repo_type='model')
# Upload sidecar for action encoding
api.upload_file(path_or_fileobj='<PATH>/aic_xvla_meta.json', path_in_repo='aic_xvla_meta.json', repo_id=repo, repo_type='model')
"
```
