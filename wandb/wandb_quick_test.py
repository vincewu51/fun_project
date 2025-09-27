import wandb, random, time

run = wandb.init(
    project="test-project-siyu",  # You can choose any name
    config={
        "learning_rate": 0.001,
        "epochs": 3,
        "batch_size": 16,
    },
)

for epoch in range(run.config.epochs):
    wandb.log({
        "epoch": epoch,
        "loss": random.random(),
        "accuracy": random.random(),
    })
    time.sleep(1)

run.finish()