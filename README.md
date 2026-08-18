Introduction
# FedLMI: A Personalized Federated Learning Framework

This repository contains the official PyTorch implementation for **FedLMI**.

---

## 📁 Repository Structure

```text
.
├── dataset/                      # Dataset directories and partition scripts
│   └── Fashion-MNIST_20client_01dir/
├── FedLMI/                       # Core algorithm implementation
│   ├── flcore/
│   │   ├── clients/              # Client-side local training logic
│   │   ├── servers/              # Server-side aggregation algorithms
│   │   └── trainmodel/           # Neural network model architectures
│   ├── utils/                    # Helper functions, metrics, and data loaders
│   └── main.py                   # Main entry point for experiments
├── other_experimental_results/   # Experimental logs and convergence analysis reports
└── environment.yml               # Conda environment configuration file
