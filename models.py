"""
models.py
---------
CNN architectures for traffic sign classification on the Kaggle Road Sign
Detection dataset (4 classes: trafficlight, stop, speedlimit, crosswalk).

Models (in order of complexity):
    ConvNet           - Baseline: strided convolutions, no pooling
    ConvNetMaxPooling - Adds 2x2 max pooling after each conv block
    ConvNetBN         - Adds custom batch normalization
    ConvNetDropout    - Adds inverted dropout as a regularizer
    ResNet            - Full residual network with skip connections

Training utilities:
    train()           - Supervised training loop with per-epoch loss tracking
    val()             - Validation loop returning average loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Training Utilities
# ──────────────────────────────────────────────────────────────────────────────

def val(model, val_data_loader, criterion, device):
    """
    Evaluate the model on a validation set and return the average loss.

    Args:
        model (nn.Module):              Model to evaluate.
        val_data_loader (DataLoader):   Validation set DataLoader.
        criterion (nn.Module):          Loss function.
        device (torch.device):          Device to run evaluation on.

    Returns:
        float: Average validation loss across all batches.
    """
    val_running_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

    model.train()
    return val_running_loss / len(val_data_loader)


def train(model, data_loader, val_data_loader, criterion, optimizer, epochs, device):
    """
    Train a model for a fixed number of epochs, logging train and validation
    loss at each epoch.

    Args:
        model (nn.Module):              Model to train.
        data_loader (DataLoader):       Training set DataLoader.
        val_data_loader (DataLoader):   Validation set DataLoader.
        criterion (nn.Module):          Loss function.
        optimizer (Optimizer):          Parameter optimizer.
        epochs (int):                   Number of training epochs.
        device (torch.device):          Device to run training on.

    Returns:
        tuple: (train_loss_arr, val_loss_arr) — per-epoch loss arrays.
    """
    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(data_loader)
        train_loss_arr.append(epoch_train_loss)

        epoch_val_loss = val(model, val_data_loader, criterion, device)
        val_loss_arr.append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train Loss: {epoch_train_loss:.4f}  "
            f"Val Loss: {epoch_val_loss:.4f}"
        )

    print("Training complete.")
    return train_loss_arr, val_loss_arr


# ──────────────────────────────────────────────────────────────────────────────
# Model 1: Baseline CNN
# ──────────────────────────────────────────────────────────────────────────────

class ConvNet(nn.Module):
    """
    Baseline CNN using strided convolutions for spatial downsampling.

    Architecture:
        conv1: 3 → 4 channels,  kernel 3, stride 2, padding 1
        conv2: 4 → 16 channels, kernel 3, stride 2, padding 1
        conv3: 16 → 32 channels, kernel 3, stride 2, padding 1
        fc1:   2048 → 1024
        fc2:   1024 → num_classes

    ReLU activations follow each conv and the first FC layer.
    Input images are expected to be 3×64×64 (RGB, normalized).
    """

    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=4,  kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4,  out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────────────
# Model 2: CNN with Max Pooling
# ──────────────────────────────────────────────────────────────────────────────

class ConvNetMaxPooling(nn.Module):
    """
    Extends ConvNet by adding 2×2 max pooling (stride 2) after each conv block.

    Max pooling reduces spatial dimensions more aggressively than strided
    convolutions alone, producing a more compact feature representation before
    the fully connected layers.
    """

    def __init__(self, num_classes=4):
        super(ConvNetMaxPooling, self).__init__()
        self.conv1 = nn.Conv2d(3,  4,  kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4,  16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────────────
# Custom Batch Normalization (from scratch)
# ──────────────────────────────────────────────────────────────────────────────

class BatchNormalization(nn.Module):
    """
    Batch normalization implemented from first principles without nn.BatchNorm2d.

    During training:
        - Computes batch mean and variance over (B, H, W) per channel.
        - Normalizes activations and applies learnable scale (gamma) and
          shift (beta) parameters.
        - Updates exponential moving averages of mean and variance for
          use during inference.

    During inference:
        - Uses stored running statistics instead of batch statistics,
          ensuring deterministic outputs.

    Args:
        num_features (int): Number of channels (C) in the input feature map.
        eps (float):        Numerical stability constant. Default: 1e-5.
        momentum (float):   EMA momentum for running statistics. Default: 0.1.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weights = nn.Parameter(torch.ones(num_features))
        self.bias    = nn.Parameter(torch.zeros(num_features))
        # Buffers move with the model to the correct device automatically
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input of shape (B, C, H, W).

        Returns:
            Tensor: Normalized output of shape (B, C, H, W).
        """
        if self.training:
            mean     = x.mean(dim=[0, 2, 3], keepdim=True)
            variance = x.var( dim=[0, 2, 3], keepdim=True, unbiased=False)
            x_norm   = (x - mean) / torch.sqrt(variance + self.eps)
            gamma    = self.weights.view(1, self.num_features, 1, 1)
            beta     = self.bias.view(   1, self.num_features, 1, 1)
            x = gamma * x_norm + beta
            # Update running statistics via EMA
            self.running_mean = self.momentum * mean.view(-1)     + (1 - self.momentum) * self.running_mean
            self.running_var  = self.momentum * variance.view(-1) + (1 - self.momentum) * self.running_var
        else:
            mean     = self.running_mean.view(1, self.num_features, 1, 1)
            variance = self.running_var.view( 1, self.num_features, 1, 1)
            x_norm   = (x - mean) / torch.sqrt(variance + self.eps)
            gamma    = self.weights.view(1, self.num_features, 1, 1)
            beta     = self.bias.view(   1, self.num_features, 1, 1)
            x = gamma * x_norm + beta
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Model 3: CNN with Batch Normalization
# ──────────────────────────────────────────────────────────────────────────────

class ConvNetBN(nn.Module):
    """
    Extends ConvNetMaxPooling with custom batch normalization after each conv.

    Block ordering: conv → BN → ReLU → pool
    Uses the custom BatchNormalization module above, not nn.BatchNorm2d.
    """

    def __init__(self, num_classes=4):
        super(ConvNetBN, self).__init__()
        self.conv1 = nn.Conv2d(3,  4,  kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4,  16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = BatchNormalization(4)
        self.bn2   = BatchNormalization(16)
        self.bn3   = BatchNormalization(32)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────────────
# Custom Dropout (from scratch)
# ──────────────────────────────────────────────────────────────────────────────

class CustomDropout(nn.Module):
    """
    Inverted dropout implemented from scratch.

    At training time, each activation is independently zeroed with probability
    `p`. Surviving activations are scaled by 1 / (1 - p) to preserve the
    expected magnitude of the output — no rescaling needed at inference time.

    At inference time, the layer is a pass-through identity.

    Args:
        p (float): Dropout probability. Must be in [0, 1). Default: 0.5.
    """

    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            x = x * mask / (1.0 - self.p)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Model 4: CNN with Dropout
# ──────────────────────────────────────────────────────────────────────────────

class ConvNetDropout(nn.Module):
    """
    Extends ConvNet with custom inverted dropout (p=0.5) applied after the
    first fully connected layer to reduce overfitting.
    """

    def __init__(self, num_classes=4):
        super(ConvNetDropout, self).__init__()
        self.conv1   = nn.Conv2d(3,  4,  kernel_size=3, stride=2, padding=1)
        self.conv2   = nn.Conv2d(4,  16, kernel_size=3, stride=2, padding=1)
        self.conv3   = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(32 * 8 * 8, 1024)
        self.dropout = CustomDropout(p=0.5)
        self.fc2     = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────────────
# Model 5: ResNet
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    A single residual block with a learnable skip connection.

    Main branch: conv1 (3×3) → BN → ReLU → conv2 (3×3) → BN → conv3 (3×3)
    Skip branch: 1×1 projection conv when input/output channels differ,
                 otherwise nn.Identity.
    Output:      ReLU(main + skip)

    The 1×1 projection ensures that channel dimensions always match before
    the elementwise addition, regardless of the stride or channel expansion.

    Args:
        in_channel (int):    Number of input channels.
        interm_channel (int): Number of channels after conv1 (intermediate).
        out_channel (int):   Number of output channels.
        stride (int):        Stride applied to conv1 and the skip projection. Default: 1.
    """

    def __init__(self, in_channel, interm_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,    interm_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(interm_channel)
        self.conv2 = nn.Conv2d(interm_channel, out_channel,   kernel_size=3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel,    out_channel,   kernel_size=3, stride=1,      padding=1, bias=False)
        self.relu  = nn.ReLU()

        # Projection shortcut: align dimensions when channels or stride changes
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        return self.relu(out + identity)


class ResNet(nn.Module):
    """
    Residual network built from stacked ResidualBlocks.

    Architecture:
        Stem:    7×7 conv → BN → ReLU → 3×3 MaxPool
        Layer 1: block_layer(num_blocks, layer1_channel → layer2_channel)
        Layer 2: block_layer(num_blocks, layer2_channel → out_channel)
        Head:    AdaptiveAvgPool(1×1) → Flatten → Linear(num_classes)

    Depth and width are fully configurable via constructor arguments.

    Args:
        num_blocks (int):      Number of ResidualBlocks per block layer.
        layer1_channel (int):  Output channels of the stem / input to layer 1.
        layer2_channel (int):  Output channels of layer 1 / input to layer 2.
        out_channel (int):     Output channels of layer 2.
        num_classes (int):     Number of classification targets. Default: 4.
    """

    def __init__(self, num_blocks, layer1_channel, layer2_channel, out_channel, num_classes=4):
        super(ResNet, self).__init__()
        self.first = nn.Sequential(
            nn.LazyConv2d(layer1_channel, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._block_layer(num_blocks, layer1_channel, layer2_channel)
        self.layer2 = self._block_layer(num_blocks, layer2_channel, out_channel)
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        )

    def _block_layer(self, num_blocks, in_channel, out_channel):
        """
        Stack `num_blocks` ResidualBlocks into a sequential layer.

        The first block uses stride=2 when the channel count changes to
        spatially downsample while expanding feature depth. Subsequent
        blocks use stride=1.
        """
        first_stride = 2 if in_channel != out_channel else 1
        blocks = [
            ResidualBlock(in_channel, out_channel, out_channel, stride=first_stride)
        ]
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(out_channel, out_channel, out_channel, stride=1)
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.last(x)
