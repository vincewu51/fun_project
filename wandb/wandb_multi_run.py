import wandb
import random
import time

# Define different hyperparameter combinations
hyperparams_list = [
    {"learning_rate": 0.01, "batch_size": 16, "epochs": 3},
    {"learning_rate": 0.001, "batch_size": 16, "epochs": 3},
    {"learning_rate": 0.005, "batch_size": 32, "epochs": 3},
]

for params in hyperparams_list:
    run = wandb.init(
        project="test-project-siyu",
        name=f"lr{params['learning_rate']}_bs{params['batch_size']}",
        config=params
    )

    # Dummy training loop
    for epoch in range(run.config.epochs):
        loss = random.random()
        acc = random.random()

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "accuracy": acc,
        })
        time.sleep(1)

    run.finish()