# FedLMI: Personalized Federated Learning Framework

## Introduction

This repository contains the PyTorch implementation of the **FedLMI** framework for personalized federated learning. FedLMI focuses on mitigating statistical heterogeneity across distributed clients through personalized local model optimization and adaptive aggregation.

---

## Datasets and Environments

For demonstration, we provide the **Fashion-MNIST** dataset partitioned under the Dirichlet Non-IID setting (Dir(0.1) for 20 clients) in the `dataset/` directory.

You can set up the required Python environment using Conda:

```bash
conda env create -f environment.yml
conda activate <your_env_name>
```

---

## System Overview

```text
.
├── dataset/
│   └── Fashion-MNIST_20client_01dir/    : Sample partitioned Non-IID dataset.
├── FedLMI/
│   ├── flcore/
│   │   ├── clients/clientFedLMI.py      : Client-side local training and personalization logic.
│   │   ├── servers/serverFedLMI.py      : Server-side model aggregation and coordination.
│   │   └── trainmodel/models.py         : Backbone neural network architectures.
│   ├── utils/
│   │   ├── HWVA.py                      : Implementation of weighting and aggregation utilities.
│   │   └── data_utils.py                : Data loading and batch generation utilities.
│   └── main.py                          : Main execution script with hyperparameter configurations.
├── other_experimental_results/          : Detailed Dirichlet analysis reports and evaluation plots.
└── environment.yml                      : Conda environment dependencies.
```

---

## Some Experimental Results

The table below presents the average test accuracy (%) comparison between our method and baseline personalized federated learning approaches under two Non-IID settings with a 50% client participation rate:

| Setting | Practical Non-IID | | | Pathological Non-IID | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Datasets** | **F-MNIST** | **CIFAR-10** | **CIFAR-100** | **F-MNIST** | **CIFAR-10** | **CIFAR-100** |
| FedAvg | 85.74 | 59.47 | 31.55 | 79.37 | 56.49 | 26.29 |
| FedProx | 85.70 | 59.59 | 31.77 | 77.31 | 56.27 | 26.44 |
| Per-FedAvg | 95.80 | 85.05 | 34.49 | 99.34 | 87.41 | 53.71 |
| pFedMe | 97.61 | 82.17 | 45.83 | 99.37 | 81.58 | 53.33 |
| FedFomo | 97.20 | 88.29 | 44.49 | 99.41 | 90.64 | 63.39 |
| FedBABU | 97.32 | 91.03 | 49.71 | 99.40 | 90.71 | 61.84 |
| FedProto | 97.05 | 89.28 | 48.73 | 99.43 | 89.43 | 65.27 |
| FedAMP | 97.29 | 88.94 | 46.92 | 99.46 | 88.51 | 62.87 |
| FedRep | 97.62 | 90.53 | 49.72 | 99.55 | 90.53 | 66.03 |
| FedALA | 97.78 | 90.84 | 55.90 | 99.58 | 91.23 | 66.20 |
| FedMCA | 97.83 | 91.76 | 59.32 | 99.61 | 91.79 | 72.33 |
| **FedLMI (ours)** | **97.86** | **92.19** | **64.37** | **99.63** | **92.65** | **75.45** |

*(More experimental results can be found under the other_experimental_results/ directory and in the paper of this work)*

---

## Training and Evaluation

To reproduce the experiments on the Fashion-MNIST dataset, run the following commands:

```bash
cd FedLMI
python main.py -hn file_name -data Fashion-MNIST_20client_01dir -nb 10 -nc 20 -gr 100
```
