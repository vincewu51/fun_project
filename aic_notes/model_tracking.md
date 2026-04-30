# AIC X-VLA Model Tracking

## Version Overview

| Version | HF Repo | Enc | Data | Method | Iters | Train Time | Chunk | Replan | Val Pos | Score | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v1** | [`v1-lora-full-delta`](https://huggingface.co/siyulw2025/v1-lora-full-delta) | delta | 270K | LoRA, lr=5e-4 | 40K | ~45 min | 30 | 15 | - | 63/300 | 04-26 |
| **v2** | [`v2-lora-full-abs`](https://huggingface.co/siyulw2025/v2-lora-full-abs) | abs | 270K | LoRA, lr=5e-4 | 20K | ~40 min | 30 | 15 | - | 63/300 | 04-27 |
| **v3** | [`v3-lora-sub-abs`](https://huggingface.co/siyulw2025/v3-lora-sub-abs) | abs | 103K | LoRA, lr=5e-4, coef=0.1 | 50K | ~1.7h | 30 | 15 | 1.6cm | - | 04-29 |
| **v4** | local only | abs | 270K | Full FT + frozen VLM | 20K | ~30 min | 30 | 15 | - | - | 04-29 |
| **v5** | `v5-lora-r32-full-delta-velmode` (planned) | delta | 270K | LoRA r32, lr=5e-4, coef=1.0 | 30K | ~5h (4080 SUPER) | 30 | 15 | - | TBD | 04-30 |
| **v6** | `v6-fullft-full-abs-jointlr` (planned) | abs | 270K | Full FT joint, vlm_lr=1e-5 / head_lr=1e-4 | 20K | ~6h (5080) | 30 | 15 | - | TBD | 04-30 |

**Defaults:** action chunk=30 steps (1.5s), replan=15, diffusion steps=10, control=pose

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

### v5 — `v5-lora-r32-full-delta-velmode` (planned, 4080 SUPER, 16 GB Ada)
- **Hypothesis**: tier-3 (insertion) is bimodal across all v1/v2 runs; LoRA r=8 is capacity-limited for the precise sub-task. Bumping rank (4× trainable params) + switching eval to velocity control mode (delta outputs are small-action, pose mode integrates them and underdrives insertion) directly attacks the tier-3 ceiling.
- **Action encoding**: delta (delta variance run hit 68.2 once vs abs ceiling 44.9 — higher upside).
- **Training**: LoRA, **rank=32, alpha=64**, lr=5e-4, **learning_coef=1.0** (do NOT repeat v3's coef=0.1 mistake), bf16, 30K iters, save_interval=1500 (20 ckpts to sweep).
- **Data**: Full 270K frames.
- **Eval control mode**: `AIC_XVLA_CMD_MODE=velocity` (NOT pose). Replan=15, timeout=180.
- **Wall time**: ~5h training on RTX 4080 SUPER (Ada, bf16 native).
- **Pre-flight checks**:
  - Confirm host XVLA conda env: `conda env list | grep XVLA`. If missing, `conda env create -f ~/workspace/X-VLA/environment.yml -n XVLA && pip install -e ~/workspace/aic/aic_utils/aic_xvla`.
  - Check `~/workspace/X-VLA/` is intact (read-only by convention) and HF cache has `2toINF/X-VLA-Pt`: `ls ~/.cache/huggingface/hub/models--2toINF--X-VLA-Pt`.
  - Confirm full meta exists: `ls /home/yifeng/aic_xvla_data/full/aic_train_meta.json`. If missing, regenerate with `python -m aic_xvla.build_meta --parquet-glob '<DATA>/episodes/*/data.parquet' --image-root <DATA> --instruction "insert the SFP cable into the port" --fps 20 --out /home/yifeng/aic_xvla_data/full/aic_train_meta.json`.
  - LoRA rank+alpha override: X-VLA's `peft_train.py` reads them from CLI `--lora_r` / `--lora_alpha` (or env vars in newer X-VLA HEAD — `grep -nE "lora_r|lora_alpha" ~/workspace/X-VLA/peft_train.py` to confirm flag names). If neither exists, edit X-VLA peft config in a *throwaway branch only* — never modify the upstream X-VLA worktree directly.
  - Sidecar must be written next to ckpt: `{"action_encoding":"delta","mode":"peft","wandb_run_name":"v5-lora-r32-full-delta-velmode","lora_r":32,"lora_alpha":64}`.
- **Launch (single line, paste into terminal on training host)**:
  ```bash
  cd ~/workspace/aic/.claude/worktree-sub3-0427-abs && \
  conda activate XVLA && \
  export XVLA_REPO=~/workspace/X-VLA && \
  export PYTHONPATH=$XVLA_REPO:$PWD/aic_utils/aic_xvla && \
  export CUDA_VISIBLE_DEVICES=0 && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export AIC_XVLA_ACTION_ENCODING=delta && \
  OUT=/home/yifeng/aic_xvla_data/v5_lora_r32_delta/ckpt && mkdir -p $OUT && \
  accelerate launch --num_processes 1 --mixed_precision bf16 -m aic_xvla.train \
      --mode peft --models 2toINF/X-VLA-Pt \
      --train_metas_path /home/yifeng/aic_xvla_data/full/aic_train_meta.json \
      --output_dir $OUT \
      --lora_r 32 --lora_alpha 64 \
      --batch_size 1 --learning_rate 5e-4 \
      --iters 30000 --save_interval 1500 \
      --wandb-project aic-xvla --wandb-run-name v5-lora-r32-full-delta-velmode
  ```
- **OOM fallback**: if 4080 SUPER 16 GB OOMs at r=32, drop to r=16/alpha=32 (still 2× v1/v2 capacity). Don't reduce batch — already 1.

### v6 — `v6-fullft-full-abs-jointlr` (planned, 5080 — separate machine)
- **Hypothesis**: full FT with VLM trained jointly is the *paper's own recipe* for X-VLA fine-tuning and has never been run cleanly on aic. v4 attempted full FT but with a buggy freeze; this is the unexplored arm with the highest theoretical upside. Differential LR (VLM low, action expert high) prevents catastrophic VLM drift while letting the head adapt fast.
- **Action encoding**: absolute (cleaner gradient signal for full FT than delta; delta benefits more from LoRA's residual bias).
- **Training**: **Full FT, no freezing** (do NOT pass any `--freeze_*` flag), differential LR via param groups: VLM=1e-5, action expert=1e-4. bf16, 20K iters, save_interval=1000.
- **Data**: Full 270K frames (same meta as v5).
- **Eval control mode**: pose (matches v2 baseline; full FT capacity makes velocity mode unnecessary). Replan=15, timeout=180.
- **Wall time**: ~6h training on RTX 5080 (Blackwell, sm_120).
- **Pre-flight checks (5080-specific)**:
  - **Blackwell needs torch built for sm_120.** Stock XVLA env ships torch 2.1; this won't run on 5080. Install torch ≥ 2.5 cu128 nightly *into a clone of the XVLA env*: `conda create -n XVLA-bw --clone XVLA && conda activate XVLA-bw && pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`. Do NOT modify the original XVLA env.
  - Verify: `python -c "import torch; print(torch.cuda.get_device_capability(0))"` should print `(12, 0)` for sm_120.
  - HF cache + X-VLA repo + aic_xvla package mirrored to 5080 host. Use `rsync -av ~/workspace/X-VLA <5080-host>:~/workspace/` and `rsync -av ~/.cache/huggingface/hub/models--2toINF--X-VLA-Pt <5080-host>:~/.cache/huggingface/hub/`.
  - Mirror data: `rsync -av /home/yifeng/aic_xvla_data/full <5080-host>:/home/yifeng/aic_xvla_data/`.
  - **Validate freeze is OFF before kicking off 20K**: dump `model.named_parameters()` after init, assert >= 95% have `requires_grad=True`. v4 silently re-enabled grads despite freeze flags — make sure you assert the *opposite* for v6 (everything trainable).
  - Differential LR: X-VLA's `train.py` accepts `--vlm_lr` / `--head_lr` if present (`grep -n "vlm_lr\|param_groups" ~/workspace/X-VLA/train.py`). If absent, pass a single `--learning_rate 1e-4` and add a 2-group optimizer override via small patch in `aic_xvla/train.py` (one-screen change; do NOT touch X-VLA upstream). Decision tree: prefer official flag → fallback to in-wrapper patch.
  - Sidecar: `{"action_encoding":"absolute","mode":"full","wandb_run_name":"v6-fullft-full-abs-jointlr","vlm_lr":1e-5,"head_lr":1e-4}`.
- **Launch (run on 5080 host after pre-flight)**:
  ```bash
  cd ~/workspace/aic/.claude/worktree-sub3-0427-abs && \
  conda activate XVLA-bw && \
  export XVLA_REPO=~/workspace/X-VLA && \
  export PYTHONPATH=$XVLA_REPO:$PWD/aic_utils/aic_xvla && \
  export CUDA_VISIBLE_DEVICES=0 && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export AIC_XVLA_ACTION_ENCODING=absolute && \
  OUT=/home/yifeng/aic_xvla_data/v6_fullft_abs/ckpt && mkdir -p $OUT && \
  accelerate launch --num_processes 1 --mixed_precision bf16 -m aic_xvla.train \
      --mode full --models 2toINF/X-VLA-Pt \
      --train_metas_path /home/yifeng/aic_xvla_data/full/aic_train_meta.json \
      --output_dir $OUT \
      --batch_size 1 --learning_rate 1e-4 \
      --vlm_lr 1e-5 --head_lr 1e-4 \
      --iters 20000 --save_interval 1000 \
      --gradient_checkpointing \
      --wandb-project aic-xvla --wandb-run-name v6-fullft-full-abs-jointlr
  ```
- **OOM fallback (16 GB Blackwell)**: full FT of ~890M params on bf16 ≈ 7 GB weights + optimizer states; tight. If OOM: add `--gradient_accumulation_steps 4` (effective batch 4 over 4 micro-batches), then deepspeed ZeRO-2 if still tight (`accelerate config` → ZeRO-2). Do **not** reduce iters — the experiment needs the full 20K to be informative.
- **Dry-run gate**: before the 20K launch, run `--iters 100 --save_interval 50` first. Confirm: (1) loss decreases, (2) `nvidia-smi` GPU mem stable, (3) ckpt saves correctly with sidecar. Costs ~5 min, saves a wasted 6 hr if config is wrong.

### Selection protocol after both finish
- For each of v5 and v6: pick top 3 ckpts by training loss + 3 ckpts spread across the run.
- Run **5 evals per ckpt** (sim variance is wide — single eval is noise; see `~/aic_results_archive/sub3_*_run*.log` showing 30→68 swings on identical ckpt).
- Rank by **median total score** (not max — max is luck). Then sanity-check by tier-3 median (insertion is the discriminator).
- Submit best ckpt to ECR via the `submission` skill workflow (`~/workspace/aic/.claude/skills/submission/SKILL.md`).

### Sim baseline check (run BEFORE training, ~2h)
The 4/27–4/28 dockerized evals showed massive variance (30→68 on same ckpt). Before burning 8 GPU-hours, confirm sim isn't worse now:
- Pull v1 (`siyulw2025/v1-lora-full-delta`) and run 5 evals. Median should be ≥ 40. If < 20, sim regressed — fix sim before training v5/v6.
- Same for v2 (`siyulw2025/v2-lora-full-abs`) if v1 looks OK.

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
