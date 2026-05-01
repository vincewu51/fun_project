# AIC X-VLA Model Tracking

## Version Overview

| Version | HF Repo | Enc | Data | Method | Chunk | Replan | Score | Date |
|---|---|---|---|---|---|---|---|---|
| **v1** | v1-lora-full-delta | delta | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-26 |
| **v2** | xvla-absolute/ckpt-20000 | abs | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-27 |
| **v3** | v3-lora-sub-abs | abs | 103K | LoRA r=8, lr=5e-4, coef=0.1 | 30 | 15 | - | 04-29 |
| **v4** | local only | abs | 270K | Full FT + frozen VLM | 30 | 15 | - | 04-29 |
| **v5** | v5-lora-r32-sub-abs-velmode | abs | 293K (cableholder-all) | LoRA r=32, lr_coef=1.0, vel mode | 30 | 15 | trained ✓ | 04-30 |
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

### v5 — v5-lora-r32-sub-abs-velmode (trained ✓ 2026-04-30)
LoRA r=32, **absolute** encoding, vel mode at eval, learning_coef=1.0. 30K iters in 2h07m on 4080 SUPER. 20 ckpts saved, **all G2 PASS** (`pred_pos_std` 0.30 ± 0.01 once converged). Final position_loss ~0.036. W&B: https://wandb.ai/wuyifeng51/aic-xvla/runs/1tyyuw44. Detailed run journal: `v5_run_2026-04-30.md`.

Data: `siyulw2025/cableholder-all` (raw LeRobot v3.0, 1.3 GB) → flat parquet+JPG (180 ep, 293K frames, 14 GB). Original plan was sub-v2 (103K) but `siyulw2025/aic-data-sub-v2` is meta-only; trained on the full converted 293K instead.

**Pending:** G3 sim eval sweep (interactive Gazebo) at ckpts {18000, 21000, 24000, 27000, 30000} × 5 evals, then HF upload (`/home/yifeng/aic_xvla_data/v5-lora-r32-sub-abs-velmode/upload_to_hf.sh` once authed), then ECR submission of best.

**Why this recipe (vs v3 stuck-arm):** v3 used same sub data + abs encoding but `learning_coef=0.1` under-trained the VLM. v5 fixes that (coef=1.0) AND bumps LoRA rank 8→32 (4× capacity for tier-3 precision) AND switches eval to velocity (`(target−live)/dt`, error-driven, dodges controller-lag pose-integration). Three changes stacked.

### v7 — v7-lora-full-delta-short8 (ready)
Shorter horizon (8 actions), delta, replan=4. Hypothesis: less drift.

### v5 historical context — original delta+velmode plan (superseded)
LoRA r=32, ABSOLUTE encoding, full data, eval with velocity mode. Companion to v5 (delta+velmode) — isolates encoding as only difference. Branch `sl-v8-lora-r32-full-abs-velmode`, PR sl628/aic#27.

**Why absolute (not delta) with velocity:** reading `handler.py:65-80` + `X-VLA/datasets/utils.py:99-104` + `eval.py:147-148`, delta training uses commanded action at chunk start as proprio, but inference adds live TCP pos. Controller lag → inferred_target ≈ lagged_state + small_delta. Pairing delta with velocity (`(target−live)/dt`) compounds the lag (small_delta → small_twist → freeze). Absolute sidesteps this entirely.

**Code/scripts (PR sl628/aic#28):**
- `aic_utils/aic_xvla/aic_xvla/train.py`: new CLI flags `--lora_r`, `--lora_alpha`, `--cmd-mode-recommended`, `--norm-check-fixture`, `--norm-check-threshold`. Monkey-patches `peft_train.LoraConfig` so X-VLA upstream stays untouched.
- `aic_utils/aic_xvla/aic_xvla/save_norm_check.py` (new): post-save hook runs `AICXVLAPolicy.predict` on `aic_data_one_ep`, asserts `pred_action[:,:3].std() > 0.005`, writes `<ckpt>/norm_check.json`. Catches "loss decreases but action collapses" silent failure before 30K iters wasted.
- `aic_utils/aic_xvla/aic_xvla/eval.py`: warns at policy init if runtime `AIC_XVLA_CMD_MODE` differs from sidecar `cmd_mode_recommended`.
- `scripts/train_v5-lora-r32-sub-abs-velmode.sh`: launch script. Reads `AIC_XVLA_DATA_ROOT`, refuses to start if missing.
- `scripts/eval_v5.sh`: eval helper. Sets velocity mode + tees log.

**G1 (code-patch sanity) PASSED 2026-04-30 09:39:** 4-iter dry train on `aic_data_one_ep`. Trainable=35.9M (≈4× v1/v2). Both ckpts pass save-hook (`pred_pos_std` 0.155, 0.238 ≫ 0.005). Sidecar correct.

**As-shipped (2026-04-30):** branch renamed to `sl-v5-lora-r32-sub-abs-velmode`, PR #28. Data is the converted-from-`cableholder-all` 293K (sub-v2 was meta-only). Training completed 2h07m, 20 ckpts, all G2 PASS. See `v5_run_2026-04-30.md` for full journal and final summary.

**Lesson learned:** explicitly set `CUDA_VISIBLE_DEVICES=0` in outer env before any background launch on this host — TITAN Xp (sm_61, GPU 1) is unsupported by XVLA env's torch and torch will silently grab it if visible. Now also hardcoded in the launch script.

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
