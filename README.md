# VGG11 MNIST Benchmark

Training and evaluation of VGG11 architecture on MNIST dataset with generalization testing.

## Architecture

VGG11 with batch normalization adapted for MNIST:
- Input: 1×32×32 (MNIST resized from 28×28 to 32×32)
- 5 convolutional blocks with max-pooling
- 3 fully connected layers with dropout (0.5)
- Output: 10 classes
- Loss: Cross-entropy

The resize to 32×32 is necessary because VGG11 has 5 max-pooling layers (stride 2), requiring input dimensions divisible by 2^5=32. The architecture reduces spatial dimensions: 32→16→8→4→2→1.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

Run individually:

```bash
# Train baseline model (5 epochs)
python3 train.py

# Generate training plots
python3 plot_training.py

# Test generalization (flips and noise)
python3 test_generalization.py

# Train with data augmentation
python3 train_augmented.py
```

## Experiments

### (a) Baseline Training
- 5 epochs on full MNIST training set
- Optimizer: Adam (lr=0.001)
- Batch size: 128
- LR schedule: StepLR (decay by 0.5 every 2 epochs)

### (b) Training Analysis
Four plots generated in `results/training_plots.png`:
1. Training accuracy vs epochs
2. Test accuracy vs epochs
3. Training loss vs epochs
4. Test loss vs epochs

### (c) Generalization Testing

#### Flip Tests
- **Horizontal flip**: Tests left-right symmetry
- **Vertical flip**: Tests top-bottom symmetry
- **Effect**: Accuracy drops significantly because MNIST digits have orientation-dependent meanings (e.g., 6↔9, left-leaning vs right-leaning digits).

#### Gaussian Noise Tests
- Variance levels: 0.01, 0.1, 1.0
- **Effect**: Higher noise variance progressively degrades accuracy by obscuring the learned features.

### (d) Data Augmentation
Retrained model with augmentations:
- Random rotation (±10°)
- Random translation (±10%)

Improves generalization, especially under noise conditions.
