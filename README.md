# Traffic Sign Classification with Convolutional Neural Networks

A PyTorch deep learning project comparing five CNN architectures on a 4-class road sign detection task that is built and trained on Google Colab Pro.

**Dataset:** [Kaggle Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) 

---

## Models

| Model | Description |
|---|---|
| `ConvNet` | Baseline: 3 strided conv layers + 2 FC layers, no pooling |
| `ConvNetMaxPooling` | Adds 2×2 max pooling after each conv block |
| `ConvNetBN` | Adds custom batch normalization (implemented from scratch) |
| `ConvNetDropout` | Adds inverted dropout (p=0.5) as a regularizer |
| `ResNet` | Full residual network with skip connections and stacked residual blocks |

All implementations are in `models.py`. `BatchNormalization` and `CustomDropout` are both built from PyTorch primitives without using `nn.BatchNorm2d` or `nn.Dropout`.

---

## How to Start

### Running in Google Colab (recommended)

1. Open `Traffic_Sign_CNN.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to GPU: click Runtime, then Change runtime type, then click A100
3. When you run all cells, the notebook will:
   - Clone this repo to access `models.py`
   - Download the dataset directly from Kaggle
   - Preprocess images into train/val/test splits
   - Train all five models and plot loss curves
   - Save predictions to `results/`

**Kaggle API token required.** When prompted, upload your `kaggle.json` file. Get it from [kaggle.com](https://www.kaggle.com), go to Your Profile, then Settings, then API, and finally Create New Token.

### Running Locally

```bash
git clone https://github.com/Walekazam/traffic-sign-cnn-pytorch
cd traffic-sign-cnn-pytorch
pip install -r requirements.txt
jupyter notebook Traffic_Sign_CNN.ipynb
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
Pillow
pandas
kaggle
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Data Pipeline

Raw images are downloaded from Kaggle and preprocessed into `.pkl` splits by the notebook. The following transforms are applied:

```python
transforms.Resize((64, 64))
transforms.ToTensor()
transforms.Normalize([0.491, 0.488, 0.512], [0.249, 0.248, 0.234])
```

Splits: 70% train · 15% validation · 15% test. The `.pkl` files are generated at runtime and are not committed to the repo.

---

## Training Setup

All models are trained with identical settings for a fair comparison:

| Setting | Value |
|---|---|
| Optimizer | SGD (momentum=0.9) |
| Learning rate | 0.001 |
| Epochs | 40 |
| Batch size | 32 |
| Loss function | Cross-entropy |
| Hardware | Google Colab Pro (A100 GPU) |

---

## Results

- **Baseline CNN** establishes the performance floor with reasonable but slower convergence
- **MaxPooling** improves generalization by reducing spatial dimensions
- **BatchNorm** produces the smoothest loss curves and fastest convergence
- **Dropout** shows higher training loss (expected from regularization) but competitive validation performance
- **ResNet** achieves the best overall validation loss through skip connections which enable more effective gradient flow and richer feature learning

---

## Key Implementation Details

**Custom BatchNormalization**: tracks running mean/variance via EMA for inference, applies learnable γ and β per channel, handles (B, C, H, W) broadcasting manually.

**Custom Dropout**: inverted dropout: scales surviving activations by `1/(1-p)` at training time, identity at inference. No rescaling needed at test time.

**ResidualBlock**: 1×1 projection shortcut when input/output channel dimensions differ, ensuring elementwise addition is always valid regardless of stride or channel expansion.

**ResNet**: configurable depth and width via `num_blocks`, `layer1_channel`, `layer2_channel`, `out_channel` constructor arguments.
