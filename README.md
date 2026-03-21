# Traffic Sign Classification with CNNs

A PyTorch deep learning project implementing and comparing multiple convolutional neural network architectures for 4-class road sign detection, including a custom ResNet with residual blocks built from scratch.

## Overview

This project explores how architectural choices — pooling, batch normalization, dropout, and residual connections — affect model performance on an image classification task. All models are trained and evaluated on the [Kaggle Road Sign Detection dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection), which contains four classes: **Traffic Light**, **Stop**, **Speed Limit**, and **Crosswalk**.

## Models Implemented

| Model | Key Feature |
|---|---|
| `ConvNet` | Baseline 3-layer CNN with strided convolutions |
| `ConvNetMaxPooling` | Adds max pooling after each conv block for spatial downsampling |
| `ConvNetBN` | Custom batch normalization from scratch |
| `ConvNetDropout` | Custom dropout (inverted dropout) for regularization |
| `ResNet` | Full residual network with skip connections and stacked residual blocks |

## Architecture Details

**Baseline ConvNet**
- 3 convolutional layers (4 → 16 → 32 channels), kernel size 3, stride 2, padding 1
- 2 fully connected layers (1024 → 4 classes)
- ReLU activations throughout

**Custom BatchNormalization**
- Implemented from scratch using PyTorch primitives
- Tracks running mean/variance for inference mode
- Learnable scale (γ) and shift (β) parameters per channel

**Custom Dropout**
- Inverted dropout: scales activations at training time, identity at inference
- Applied after the FC layer with p=0.5

**ResNet**
- Residual blocks with 3×3 conv → BN → ReLU → 3×3 conv + skip connection
- Projection shortcut (1×1 conv) when channel dimensions change
- Two stacked block layers followed by adaptive average pooling

## Training Pipeline

The `train()` function implements the standard supervised learning loop:

```python
for epoch in range(epochs):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

- **Loss function:** Cross-Entropy Loss
- **Optimizer:** SGD / Adam (configurable)
- **Tracking:** Train and validation loss logged per epoch for learning curve analysis

## Setup & Usage

**Requirements**
```
torch
torchvision
numpy
pandas
matplotlib
Pillow
```

Install dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib Pillow
```

**Run training**
```python
from submission import ConvNet, ResNet, train
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(num_blocks=2, layer1_channel=64, layer2_channel=128, out_channel=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=20, device=device)
```

**Data loading**
```python
# Load pre-processed pickle datasets
train_dataset = load_dataset("train", base_dir)
val_dataset   = load_dataset("val", base_dir)
test_dataset  = load_dataset("test", base_dir, test_mode=True)
```

Images are pre-processed to 64×64 and normalized:
```python
transforms.Resize((64, 64))
transforms.Normalize([0.491, 0.488, 0.512], [0.249, 0.248, 0.234])
```

## Project Structure

```
├── Assignment_1.ipynb   # Full experiment notebook with analysis and plots
├── submission.py        # Model implementations and training loop
├── train_data.pkl       # Training set
├── val_data.pkl         # Validation set
├── test_data.pkl        # Test set
└── README.md
```

## Skills Demonstrated

- **PyTorch model design** — building custom `nn.Module` classes with full control over forward passes
- **Training loop engineering** — gradient zeroing, backprop, optimizer stepping, loss tracking
- **Regularization techniques** — batch normalization and dropout implemented from first principles
- **Residual networks** — skip connections with projection shortcuts for dimension matching
- **Hyperparameter analysis** — learning rate sweeps with performance curve visualization
- **Data pipeline** — custom `Dataset` and `DataLoader` for pickle-serialized image data

## Relevance to ML Engineering

This project directly demonstrates:
- Hands-on experience with **PyTorch** and deep learning model implementation
- Understanding of **CNN architectures** and the effect of architectural choices on convergence
- Implementation of **batch normalization** and **dropout from scratch** (not just using library calls)
- **Model evaluation** with train/val loss tracking and learning curve analysis
- Structured, readable Python code following ML engineering conventions
