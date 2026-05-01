# Model Logs

Shared training metadata for X-VLA model comparison dashboard. Each subdirectory is one training run, containing:

- `aic_xvla_meta.json` — action encoding, training mode, wandb run name
- `train.log` — hyperparameters + eval results (`Eval [step]: pos=X rot=Y`)

## Adding a model from any machine

```bash
# 1. Clone fun_project (once)
git clone git@github.com:vincewu51/fun_project.git ~/fun_project

# 2. After training, sync your local model logs
bash ~/aic/scripts/sync_models_to_funproject.sh
```

This copies your local `aic_xvla_meta.json` + `train.log` from `~/aic_xvla_data/` into this directory, commits, and pushes. No scp, no manual copying.

## Rebuilding the dashboard

```bash
python3 ~/aic/scripts/build_dashboard.py
```

Scans both `~/aic_xvla_data/` (local checkpoints) and this `model-logs/` directory (all machines' logs), merges by model ID, and writes `docs/model-compare/index.html`.

Dashboard: `https://vincewu51.github.io/fun_project/model-compare/`
