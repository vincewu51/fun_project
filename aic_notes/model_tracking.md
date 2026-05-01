# AIC X-VLA Model Tracking

## Version Overview

| Version | HF Repo | Enc | Data | Method | Chunk | Replan | Score | Date |
|---|---|---|---|---|---|---|---|---|
| **v1** | v1-lora-full-delta | delta | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-26 |
| **v2** | xvla-absolute/ckpt-20000 | abs | 270K | LoRA r=8, lr=5e-4 | 30 | 15 | 63/300 | 04-27 |
| **v3** | v3-lora-sub-abs | abs | 103K | LoRA r=8, coef=0.1 | 30 | 15 | - | 04-29 |
| **v5** | v5-lora-r32-sub-abs-velmode | abs | 293K | LoRA r=32, vel mode | 30 | 15 | **hover** (abs/proprio bug) | 04-30 |
| **v5b** | v5-lora-r32-sub-delta-posmode | delta | 293K | LoRA r=32, pose mode | 30 | 15 | training | 04-30 |
| **v6** | v6-fullft-full-abs-jointlr | abs | 270K | Full FT, head=1e-4/vlm=1e-5 | 30 | 15 | - | 05-01 |
| **v7** | v7-lora-full-delta-short8 | delta | 270K | LoRA r=8, num_actions=8 | 8 | 4 | -1.6 | 04-30 |
| **v8** | v8-lora-r32-full-abs-velmode | abs | 270K | LoRA r=32, vel mode | 30 | 15 | done | 04-30 |
| **v9** | v9-lora-r32-sub-abs-velmode-2stage | abs | 103K | LoRA r=32, 2-stage (freeze→full) | 30 | 15 | planned | 04-30 |

## Detailed Results

### v1
Delta, LoRA r=8. Moves but doesn't insert. Score 63/300.

### v2
Absolute, LoRA r=8. Same score 63/300.

### v3
Absolute, subsampled, low VLM LR. Robot stuck.

### v5 — v5-lora-r32-sub-abs-velmode (trained, FAILS in sim — hover)
LoRA r=32, absolute, velocity mode, coef=1.0. 30K on 4080 SUPER. G2 all passed.

**Sim outcome (2026-04-30 evening):** robot **hovers**, target_pos within 1-2 mm of live_tcp_pos every chunk, never approaches port. Same eval pipeline that produces correct motion on v1 (delta). Tested ckpts 10500/21000/28500/30000 — all hover identically. Velocity vs pose mode and merge_and_unload vs raw PeftModel both produce identical static output → bug is in the model, not the eval path.

**Root cause — absolute encoding has a fatal train/inference proprio mismatch:**
- Training proprio comes from the parquet's `action` column (commanded action at chunk start) — see `handler.py:_build_abs_trajectory` + `X-VLA/datasets/utils.py:action_slice` line 99 (`proprio = abs_traj[0]`).
- Inference proprio comes from live TCP pose — `_state_to_proprio(state[0:3])` reads `controller_state.tcp_pose.position`.
- Live and command differ by controller lag (mm to ~1 cm).
- For **delta** encoding the proprio cancels out of both training loss target AND inference reconstruction (eval adds proprio back). Lag is invisible.
- For **absolute** encoding the model learns `f(images, commanded_proprio) → commanded_target`, where in training they're tightly coupled (both from `action` column). Fed live TCP at inference, the model emits ≈ live TCP shifted slightly along the demonstrated direction. No correction layer. → hover.

**v5 in original plan was paired velocity+abs as "the clean pairing" — that reasoning was wrong.** Velocity at inference computes `(target − live)/dt`; with `target ≈ live`, velocity twist is also near-zero. Absolute encoding's proprio mismatch beats out any control-mode choice.

**Lesson:** for X-VLA on aic, **delta encoding is required at this controller's lag**. Absolute would only work if inference fed the *commanded* TCP back as proprio (i.e., ROS open-loop replay) which we don't do.

### v5b — v5-lora-r32-sub-delta-posmode (training, fix for v5)
Re-run of v5 with **delta** encoding (and pose mode in sidecar — v1+pose is the proven 63/300 combo). Same data, same r=32, same lr_coef=1.0, same 30K iters, same save_interval=1500. Only `AIC_XVLA_ACTION_ENCODING=delta` and `--cmd-mode-recommended pose` differ from v5-abs. Launched 2026-04-30 20:44 PT, ETA ~2h. W&B: https://wandb.ai/wuyifeng51/aic-xvla/runs/knze0tbj.

If v5b clears v1's 63/300 baseline, it confirms the capacity bump (r=32) and full VLM LR (coef=1.0) are improvements over v1 — we just had to use the right encoding.

### v6 — v6-fullft-full-abs-jointlr
Full FT, abs encoding, joint LR. First non-LoRA arm; tests whether updating the VLM jointly (vs LoRA's frozen-VLM bias) helps tier-3 insertion. Branch/worktree `yf_v6-fullft-full-abs-jointlr`. Compute: Velda `anycloud-a100-1` pool (not 5080 — A100 keeps stock `xvla-stable` env, no Blackwell torch nightly needed). Data: `siyulw2025/cableholder-all` HF dataset → `~/aic_xvla_data/cableholder-all/` via `convert_lerobot_to_xvla.py`. Joint LR uses X-VLA's existing param-group split (`--learning_rate 1e-4 --learning_coef 0.1` → vlm=1e-5, head=1e-4) — no upstream patch needed. 20K iters, save_interval=1000.


Shorter horizon (8 actions), delta. Less open-loop drift.

### v8 — v8-lora-r32-full-abs-velmode (training)
LoRA r=32, absolute, velocity mode, full data. 144 MB. Loss 0.015 at 30K.

### v9 — v9-lora-r32-sub-abs-velmode-2stage (planned)
**Goal:** follow the X-VLA fine-tuning recipe v1–v8 skipped. Same data/encoding/control as v5, but train in two stages per `peft_train.py:166-174`.

**Why:** during `step < freeze_steps`, the trainer sets `vlm` and `transformer_core` LR to 0 and only updates `soft_prompts` + `action_heads` (the embodiment-specific bits). v5 ran `freeze_steps=0, warmup_steps=50, learning_coef=1.0` — soft prompts and backbone moved together from step 0, which is the opposite of the paper's recipe. Hypothesis: warming up the soft prompt against a frozen pretrained backbone first gives a cleaner adaptation signal before LoRA touches the core.

**Setup (delta vs v5):**
- `--freeze_steps 2000` (v5: 0). Stage-1 trains soft prompt + action heads only.
- `--warmup_steps 2000` (v5: 50). Linear warmup AFTER unfreeze, per `linear_warmup_cosine(step, freeze_steps, warmup_steps, ...)`.
- `--learning_coef 0.1` (v5: 1.0). Matches the README's recommended VLM/soft-prompt LR ratio.
- `--iters 30000`, LoRA r=32/α=64, lr=5e-4, abs encoding, velocity cmd mode — all identical to v5 so the 2-stage recipe is the only variable.
- Data: `aic-data-sub-v2` (103K), same as v5.

**Expected schedule:** steps 0–2000 soft-prompt-only at lr=5e-5; steps 2000–4000 full-model warmup ramp; 4000–30000 cosine decay to 5% of base.

**Gates (reuse v5 infra):** G1 dry train on `aic_data_one_ep`. G2 save-hook `pred_pos_std > 0.005` at every save. G3 sim dry-run at ckpt-3000 (note: this is ckpt-1000 INTO stage-2 — earlier sim check than v5). Auto-fallback: if stage-1 loss doesn't drop below stage-1 v5-equivalent (loss at iter 2000), abort and re-evaluate `freeze_steps`.

**Success criterion:** median total ≥ v5 median + 5. If equal/worse, the soft-prompt-first recipe doesn't help on this embodiment and we stick with `freeze_steps=0` going forward.

### v7 — v7-lora-full-delta-short8
Shorter horizon (8 actions), delta. Score -1.6 (contact penalty). Moves better, reaches 9cm, but collides with task board.

## Naming Convention
Format: v{number}-{method}-{data}-{encoding}[-{tag}]

Must match: HF repo, W&B run name, output dir, sidecar, script filename, eval log.

## Control Mode
| Mode | Stiffness (N/m) | Damping (Ns/m) |
|---|---|---|
| pose | [75,75,75,75,75,75] | [35,35,35,35,35,35] |
| velocity | [100,100,100,50,50,50] | [40,40,40,15,15,15] |
