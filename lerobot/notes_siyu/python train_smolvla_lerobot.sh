  python train_smolvla_lerobot.py \
    --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
    --max_steps 100000 \
    --save_steps 5000 \
    --batch_size 16 \
    --chunk_size 50 \
    --wandb_project "b1k-smolvla-task0006"

  python train_smolvla_lerobot.py \
    --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
    --max_steps 100000 \
    --save_steps 5000 \
    --batch_size 16 \
    --chunk_size 50 \
    --filter_state \
    --wandb_project "b1k-smolvla-task0006"

python train_smolvla_lerobot.py \
    --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
    --filter_state \
    --batch_size 16 \
    --max_steps 100000 \
    --save_steps 5000 \
    --chunk_size 50 \
    --num_workers 0 \
    --video_tolerance 0.15 \
    --lr 1e-4 \
    --wandb_project "b1k-smolvla-task0006"

python train_smolvla_lerobot.py \
    --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
    --filter_state \
    --batch_size 32 \
    --max_steps 100000 \
    --save_steps 5000 \
    --chunk_size 50 \
    --num_workers 0 \
    --gpu_id 0 \
    --lr 1e-4 \
    --video_tolerance 0.15 \
    --wandb_project "b1k-smolvla-task0006"


python train_smolvla_lerobot.py \
    --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
    --filter_state \
    --batch_size 4 \
    --max_steps 100000 \
    --save_steps 200 \
    --keep_last_n_checkpoints 10 \
    --chunk_size 50 \
    --num_workers 0 \
    --lr 1e-4 \
    --video_tolerance 0.15 \
    --wandb_project "b1k-smolvla-task0006"
