# Physics-Informed Neural Networks for PDEs

This repository contains the implementation and experiments for comparing Unified, Modular, and Scaled Modular architectures in Physics-Informed Neural Networks (PINNs) for solving Partial Differential Equations (PDEs). The study focuses on understanding the trade-offs between these architectures in terms of convergence speed, final loss, neuron activation overlap, correlation, and interpretability.

Code fully compatible with Semi Modular networks is on the 'semimodular' branch to avoid clutering the original.

## Project Overview

This project investigates three neural network architectures applied to solving multi-component PDEs:

1. **Unified Architecture** - A single neural network solving all components of a PDE system simultaneously.
2. **Modular Architecture** - Separate networks trained for each PDE component independently.
3. **Scaled Modular Architecture** - A modular approach where sub-networks are scaled to match the parameter count of the unified network.

### PDE Systems Considered

The experiments focus on:
- **Reaction-Diffusion Systems**
- **Elastic Wave Systems**

## Code Structure

### Main Components
- `PDESystem` - Abstract base class for defining PDEs.
- `UnifiedPINN` - A fully connected neural network solving the entire PDE system.
- `ModularPINNs` - A collection of smaller networks solving each PDE component separately.
- `UnifiedVsModularComparison` - A utility class for training and comparing Unified and Modular architectures.

### Experiments

The architectures used in experiments are:

```python
experiments = [
    ((64, 64), "SmallEqual"),
    ((128, 64), "Baseline"),
    ((128, 64, 32), "MediumThreeTapered"),
    ((64, 64, 64), "MediumThreeEqual"),
    ((64, 128, 128, 64), "MidInverseTapered"),
    ((256, 256), "MediumEqual"),
    ((64, 64, 64, 64), "MediumDeepEqual"),
    ((512, 256, 128, 64), "LargeTapered"),
    ((256, 256, 256, 256), "LargeEqual"),
]
```

## Metrics for Comparison

The following metrics are used to evaluate and compare the architectures:

- **Convergence Epochs** - Number of epochs required to reach loss thresholds (e.g., `10^-4`, `10^-5`).
- **Final Loss** - The final training loss of the networks.
- **Neuron Activation Overlap** - The proportion of neurons activated in common across layers.
- **Neuron Activation Correlation** - Measures how strongly activations across layers or components influence each other.
- **Relative Training Time** - Ratio of training times between unified and modular architectures.
- **Layer-wise Sparsity and Sparsity CDF** - Amount of network inactive or redundant.

## Running the Experiments

Experiments were run on Google Collab with a T4 GPU.
