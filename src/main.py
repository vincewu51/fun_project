import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# AlexNet Model Definition for CIFAR-10
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # Modified for CIFAR-10 (32x32 images)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Adjusted classifier for CIFAR-10
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class NewtonOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1.0, damping=1e-3):
        """
        Full Newton optimizer with dense Hessian.
        Args:
            params: model parameters
            lr: learning rate scaling for Newton step
            damping: Tikhonov damping term (adds λI to Hessian for stability)
        """
        defaults = dict(lr=lr, damping=damping)
        super(NewtonOptimizer, self).__init__(params, defaults)

    def _gather_flat_params_and_grads(self, group):
        params = [p for p in group["params"] if p.grad is not None]
        grads = [p.grad for p in params]
        flat_params = parameters_to_vector(params)
        flat_grads = parameters_to_vector(grads)
        return params, flat_params, flat_grads

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single Newton update step using full Hessian."""
        if closure is None:
            raise RuntimeError(
                "Newton's method requires a closure to reevaluate the model and loss."
            )

        # Recompute loss and keep graph for Hessian
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]

            # Collect params & grads
            params = [p for p in group["params"] if p.grad is not None]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            flat_grads = parameters_to_vector(grads)
            n = flat_grads.numel()

            # Build Hessian row by row (O(n^2), only for small models!)
            H = torch.zeros(n, n, device=flat_grads.device, dtype=flat_grads.dtype)
            for i in range(n):
                gi = flat_grads[i]
                row_grads = torch.autograd.grad(gi, params, retain_graph=True)
                row = parameters_to_vector(row_grads)
                H[i, :] = row

            # Damping for stability
            H = H + damping * torch.eye(n, device=H.device)

            # Solve H Δ = g
            delta = torch.linalg.solve(H, flat_grads)

            # Update parameters
            flat_params = parameters_to_vector(params)
            new_params = flat_params - lr * delta
            vector_to_parameters(new_params, params)

        return loss


# Training function with Newton optimizer
def train_alexnet_on_cifar10_newton():
    """
    torch.cuda.OutOfMemoryError: CUDA out of memory. 
    Tried to allocate 4789210.07 GiB. GPU 0 has a total capacity of 11.90 GiB of which 10.71 GiB is free. 
    Including non-PyTorch memory, this process has 1.17 GiB memory in use. 
    Of the allocated memory 888.86 MiB is allocated by PyTorch, and 113.14 MiB is reserved by PyTorch but unallocated. 
    If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. 
    See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    num_classes = 10  # CIFAR-10 has 10 classes
    batch_size = 128
    num_epochs = 5
    learning_rate = 0.001

    # Initialize the model
    model = AlexNet(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = NewtonOptimizer(model.parameters(), lr=learning_rate)

    # Data preprocessing (adjusted for CIFAR-10)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Metrics tracking
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):

        def closure():
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # must keep graph for Hessian
            loss.backward(create_graph=True)
            return loss

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step(closure)

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Accuracy": f"{100 * correct / total:.2f}%",
                }
            )

        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix(
                    {"Accuracy": f"{100 * val_correct / val_total:.2f}%"}
                )

        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {epoch_accuracy:.2f}%, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    # Calculate total training time
    total_time = time.time() - start_time

    # Save model
    torch.save(model.state_dict(), "alexnet_cifar10_newton.pth")

    # Prepare metrics
    metrics = {
        "model": "AlexNet",
        "dataset": "CIFAR-10",
        "optimizer": "Newton",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_train_loss": train_losses[-1],
        "final_train_accuracy": train_accuracies[-1],
        "final_val_accuracy": val_accuracies[-1],
        "training_time_seconds": total_time,
        "training_time_minutes": total_time / 60,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    # Save metrics to file
    with open("training_metrics_newton.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Print final metrics
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED - FINAL METRICS (Newton Optimizer)")
    print("=" * 50)
    print(f"Model: AlexNet")
    print(f"Dataset: CIFAR-10")
    print(f"Optimizer: Newton")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print(
        f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )

    return metrics


# Training function with Adam optimizer (keeping the original for comparison)
def train_alexnet_on_cifar10_adam():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    num_classes = 10  # CIFAR-10 has 10 classes
    batch_size = 128
    num_epochs = 5
    learning_rate = 0.001

    # Initialize the model
    model = AlexNet(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data preprocessing
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Metrics tracking
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Accuracy": f"{100 * correct / total:.2f}%",
                }
            )

        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix(
                    {"Accuracy": f"{100 * val_correct / val_total:.2f}%"}
                )

        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {epoch_accuracy:.2f}%, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    # Calculate total training time
    total_time = time.time() - start_time

    # Save model
    torch.save(model.state_dict(), "alexnet_cifar10_adam.pth")

    # Prepare metrics
    metrics = {
        "model": "AlexNet",
        "dataset": "CIFAR-10",
        "optimizer": "Adam",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_train_loss": train_losses[-1],
        "final_train_accuracy": train_accuracies[-1],
        "final_val_accuracy": val_accuracies[-1],
        "training_time_seconds": total_time,
        "training_time_minutes": total_time / 60,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    # Save metrics to file
    with open("training_metrics_adam.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Print final metrics
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED - FINAL METRICS (Adam Optimizer)")
    print("=" * 50)
    print(f"Model: AlexNet")
    print(f"Dataset: CIFAR-10")
    print(f"Optimizer: Adam")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print(
        f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )

    return metrics


if __name__ == "__main__":
    print("Training with Newton optimizer...")
    newton_metrics = train_alexnet_on_cifar10_newton()

    print("\n\nTraining with Adam optimizer...")
    adam_metrics = train_alexnet_on_cifar10_adam()

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON OF OPTIMIZERS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Newton':<15} {'Adam':<15}")
    print("-" * 60)
    print(
        f"{'Final Training Loss':<25} {newton_metrics['final_train_loss']:<15.4f} {adam_metrics['final_train_loss']:<15.4f}"
    )
    print(
        f"{'Final Training Accuracy':<25} {newton_metrics['final_train_accuracy']:<15.2f} {adam_metrics['final_train_accuracy']:<15.2f}"
    )
    print(
        f"{'Final Validation Accuracy':<25} {newton_metrics['final_val_accuracy']:<15.2f} {adam_metrics['final_val_accuracy']:<15.2f}"
    )
    print(
        f"{'Training Time (seconds)':<25} {newton_metrics['training_time_seconds']:<15.2f} {adam_metrics['training_time_seconds']:<15.2f}"
    )
