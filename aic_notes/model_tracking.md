# AIC X-VLA Model Tracking

## Version Overview

| Version | HF Repo | Enc | Data | Method | Chunk | Replan | Score | Date |
|---|---|---|---|---|---|---|---|---|
| **v1** | v1-lora-full-delta | delta | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-26 |
| **v2** | xvla-absolute/ckpt-20000 | abs | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-27 |
| **v3** | v3-lora-sub-abs | abs | 103K | LoRA r=8, coef=0.1 | 30 | 15 | - | 04-29 |
| **v5** | v5-lora-r32-sub-abs-velmode | abs | 293K | LoRA r=32, vel mode | 30 | 15 | trained | 04-30 |
| **v6** | v6-fullft-full-abs-jointlr | abs | 270K | Full FT, head=1e-4/vlm=1e-5 | 30 | 15 | ready | 05-01 |
| **v7** | v7-lora-full-delta-short8 | delta | 270K | LoRA r=8, num_actions=8 | 8 | 4 | ready | 04-30 |
| **v8** | v8-lora-r32-full-abs-velmode | abs | 270K | LoRA r=32, vel mode | 30 | 15 | done | 04-30 |

## Detailed Results

### v1
Delta, LoRA r=8. Moves but doesn't insert. Score 63/300.

### v2
Absolute, LoRA r=8. Same score 63/300.

### v3
Absolute, subsampled, low VLM LR. Robot stuck.

### v5 — v5-lora-r32-sub-abs-velmode (trained)
LoRA r=32, absolute, velocity mode, coef=1.0. 30K on 4080 SUPER. G2 all passed.

### v6 — v6-fullft-full-abs-jointlr (ready)
Full FT, abs encoding, joint LR. First non-LoRA arm; tests whether updating the VLM jointly (vs LoRA's frozen-VLM bias) helps tier-3 insertion. Branch/worktree `yf_v6-fullft-full-abs-jointlr`. Compute: Velda `anycloud-a100-1` pool (not 5080 — A100 keeps stock `xvla-stable` env, no Blackwell torch nightly needed). Data: `siyulw2025/cableholder-all` HF dataset → `~/aic_xvla_data/cableholder-all/` via `convert_lerobot_to_xvla.py`. Joint LR uses X-VLA's existing param-group split (`--learning_rate 1e-4 --learning_coef 0.1` → vlm=1e-5, head=1e-4) — no upstream patch needed. 20K iters, save_interval=1000.

### v7 — v7-lora-full-delta-short8 (ready)
Shorter horizon (8 actions), delta. Less open-loop drift.

### v8 — v8-lora-r32-full-abs-velmode (trained)
LoRA r=32, absolute, velocity mode, full data. 144 MB. Loss 0.015 at 30K.

## Naming Convention
Format: v{number}-{method}-{data}-{encoding}[-{tag}]

Must match: HF repo, W&B run name, output dir, sidecar, script filename, eval log.

## Control Mode
| Mode | Stiffness (N/m) | Damping (Ns/m) |
|---|---|---|
| pose | [75,75,75,75,75,75] | [35,35,35,35,35,35] |
| velocity | [100,100,100,50,50,50] | [40,40,40,15,15,15] |
