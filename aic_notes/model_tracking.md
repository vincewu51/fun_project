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

### v8 — v8-lora-r32-full-abs-velmode (ready, blocked on 270K data)
LoRA r=32, ABSOLUTE encoding, full data, eval with velocity mode. Companion to v5 (delta+velmode) — isolates encoding as only difference. Branch `sl-v8-lora-r32-full-abs-velmode`, PR sl628/aic#27.

**Why absolute (not delta) with velocity:** reading `handler.py:65-80` + `X-VLA/datasets/utils.py:99-104` + `eval.py:147-148`, delta training uses commanded action at chunk start as proprio, but inference adds live TCP pos. Controller lag → inferred_target ≈ lagged_state + small_delta. Pairing delta with velocity (`(target−live)/dt`) compounds the lag (small_delta → small_twist → freeze). Absolute sidesteps this entirely.

**Code/scripts (PR sl628/aic#27):**
- `aic_utils/aic_xvla/aic_xvla/train.py`: new CLI flags `--lora_r`, `--lora_alpha`, `--cmd-mode-recommended`, `--norm-check-fixture`, `--norm-check-threshold`. Monkey-patches `peft_train.LoraConfig` so X-VLA upstream stays untouched.
- `aic_utils/aic_xvla/aic_xvla/save_norm_check.py` (new): post-save hook runs `AICXVLAPolicy.predict` on `aic_data_one_ep`, asserts `pred_action[:,:3].std() > 0.005`, writes `<ckpt>/norm_check.json`. Catches "loss decreases but action collapses" silent failure before 30K iters wasted.
- `aic_utils/aic_xvla/aic_xvla/eval.py`: warns at policy init if runtime `AIC_XVLA_CMD_MODE` differs from sidecar `cmd_mode_recommended`.
- `scripts/train_v8-lora-r32-full-abs-velmode.sh`: launch script. Reads `AIC_XVLA_DATA_ROOT`, refuses to start if missing.
- `scripts/eval_v8.sh`: eval helper. Sets velocity mode + tees log.

**G1 (code-patch sanity) PASSED 2026-04-30 09:39:** 4-iter dry train on `aic_data_one_ep`. Trainable=35.9M (≈4× v1/v2). Both ckpts pass save-hook (`pred_pos_std` 0.155, 0.238 ≫ 0.005). Sidecar correct.

**Launch (when 270K data lands):**
```bash
cd ~/workspace/aic/.claude/worktree-sl-v8-lora-r32-full-abs-velmode
AIC_XVLA_DATA_ROOT=/path/to/aic_xvla_data \
  bash scripts/train_v8-lora-r32-full-abs-velmode.sh
```

**Auto-fallbacks** (when running): G2 fail → swap to delta, rename to `v8-lora-r32-full-delta-velmode`. G3 fail (no motion at ckpt-3000) → swap to pose, rename to `v8-lora-r32-full-abs-posmode`. OOM at r=32 → drop to r=16/alpha=32, rename to `v8-lora-r16-full-abs-velmode`. Each rename updates all six naming locations atomically.

See: `v8_run_2026-04-30.md` for full journal.

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
