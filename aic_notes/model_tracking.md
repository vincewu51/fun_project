# AIC X-VLA Model Tracking

## Version Overview

| Version | HF Repo | Enc | Data | Method | Chunk | Replan | Score | Date |
|---|---|---|---|---|---|---|---|---|
| **v1** | v1-lora-full-delta | delta | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-26 |
| **v2** | xvla-absolute/ckpt-20000 | abs | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-27 |
| **v3** | v3-lora-sub-abs | abs | 103K | LoRA r=8, lr=5e-4, coef=0.1 | 30 | 15 | - | 04-29 |
| **v4** | local only | abs | 270K | Full FT + frozen VLM | 30 | 15 | - | 04-29 |
| **v5** | v5-lora-r32-full-delta-velmode | delta | 270K | LoRA r=32, vel mode | 30 | 15 | running | 04-30 |
| **v8** | v8-lora-r32-full-abs-velmode | abs | 270K | LoRA r=32, vel mode | 30 | 15 | ready | 04-30 |
| **v7** | v7-lora-full-delta-short8 | delta | 270K | LoRA r=8, num_actions=8 | **8** | **4** | ready | 04-30 |

## Detailed Results

### v1 — v1-lora-full-delta
Delta, LoRA r=8, full data. Moves but doesn't insert. Score 63/300.

### v2 — xvla-absolute/ckpt-20000
Absolute, LoRA r=8, full data. Same score 63/300.

### v3 — v3-lora-sub-abs
Absolute, subsampled, low VLM LR. Robot stuck.

### v4 — local only
Full FT + frozen VLM. Freeze buggy. Not tested.

### v5 — v5-lora-r32-full-delta-velmode (running on 4080)
LoRA r=32, delta, velocity mode. Running on 4080 SUPER.

### v7 — v7-lora-full-delta-short8 (ready)
Shorter horizon (8 actions), delta, replan=4. Hypothesis: less drift.

## Naming Convention
Format: `v{number}-{method}-{data}-{encoding}[-{tag}]`

Must match in all 6 locations:
1. HF repo: siyulw2025/<name>
2. W&B: --wandb-run-name <name>
3. Output dir: ~/aic_xvla_data/<name>
4. Sidecar: wandb_run_name: <name>
5. Script: scripts/train_<name>.sh
6. Eval log: ~/aic_results_archive/<name>_run*.log

## Control Mode
| Mode | Stiffness (N/m) | Damping (Ns/m) |
|---|---|---|
| pose | [75,75,75,75,75,75] | [35,35,35,35,35,35] |
| velocity | [100,100,100,50,50,50] | [40,40,40,15,15,15] |

## How to Evaluate
```bash
# Terminal 1 — Serve model
python -m aic_xvla.serve --checkpoint siyulw2025/<REPO> --single-model --action-encoding <enc>

# Terminal 2 — Policy
export PYTHONPATH=$PWD/aic_utils/aic_xvla
export AIC_XVLA_SERVER_URL=http://127.0.0.1:8010
export AIC_XVLA_CMD_MODE=pose
export AIC_XVLA_REPLAN=15
export AIC_XVLA_TASK_TIMEOUT_S=180
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_xvla.ros.RunXVLA

# Terminal 3 — Eval engine with GUI
docker run --rm --gpus all --network host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  /entrypoint.sh ground_truth:=false start_aic_engine:=true
```
| **v8** | v8-lora-r32-full-abs-velmode | abs | 270K | LoRA r=32, vel mode | 30 | 15 | ready | 04-30 |
