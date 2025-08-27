# CIFAR-10 Training Comparison: Newton vs Adam Optimizer

This project implements and compares two optimization algorithms for training a modified AlexNet on the CIFAR-10 dataset:
1. A custom Newton optimizer (second-order optimization)
2. The standard Adam optimizer (first-order optimization)

## Overview

The implementation demonstrates the practical differences between first-order and second-order optimization methods in deep learning. While Newton's method theoretically converges faster by using second-order information (Hessian matrix), it's computationally more expensive.

## Files

- `src/main.py`: Main implementation containing both optimizers and training loops
- `training_metrics_newton.json`: Metrics from Newton optimizer training (generated after run)
- `training_metrics_adam.json`: Metrics from Adam optimizer training (generated after run)
- `alexnet_cifar10_newton.pth`: Saved model weights from Newton training (generated after run)
- `alexnet_cifar10_adam.pth`: Saved model weights from Adam training (generated after run)

## Newton Optimizer Implementation

The Newton optimizer in this implementation uses a Gauss-Newton approximation for the Hessian matrix, which is more practical for neural networks than computing the exact Hessian. The update rule follows:

```
θ_{t+1} = θ_t - lr * H^{-1} * ∇L(θ_t)
```

Where:
- `θ` are the model parameters
- `lr` is the learning rate
- `H` is the Hessian approximation
- `∇L(θ_t)` is the gradient of the loss function

For computational efficiency, we use a diagonal approximation of the Gauss-Newton matrix:
```
H ≈ ∇²L ≈ J^T * J + damping*I
```

## Requirements

- Python 3.6+
- PyTorch 1.8+
- torchvision
- tqdm

## Usage

```bash
python src/main.py
```

The script will:
1. Train AlexNet on CIFAR-10 using the Newton optimizer
2. Train AlexNet on CIFAR-10 using the Adam optimizer
3. Compare and display the results of both approaches

## Expected Output

The script outputs training progress for each epoch and final metrics including:
- Final training loss
- Final training accuracy
- Final validation accuracy
- Training time for each optimizer

## Key Differences

| Aspect | Newton Optimizer | Adam Optimizer |
|--------|------------------|----------------|
| Order | Second-order (uses Hessian) | First-order (uses gradient only) |
| Convergence | Faster in theory | Slower but more stable |
| Computational Cost | Higher per iteration | Lower per iteration |
| Memory Usage | Higher | Lower |
| Implementation Complexity | More complex | Simpler |

## Notes

- The Newton optimizer uses damping to ensure numerical stability
- Training with Newton's method may take longer per epoch due to additional computations
- Results may vary based on hyperparameter tuning
- For production use, Adam is often preferred due to its robustness and efficiency

## References

1. [Newton's Method in Optimization](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)
2. [Adam Optimizer](https://arxiv.org/abs/1412.6980)
3. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
4. [Gauss-Newton Method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)

```<|im_end|>
