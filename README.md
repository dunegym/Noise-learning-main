# Noise Learning

This project implements a deep learning approach for instrumental noise modeling and removal in spectroscopy data.

## Features

- UNet-based architecture with attention mechanisms for noise removal
- Support for both Raman spectroscopy (Horiba LabRAM) and other spectroscopy instruments
- Training from scratch or resuming from checkpoints
- Testing and prediction capabilities

## Training Modes

We've unified the pretraining and training functionality with flexible options:

1. **Train from scratch**: Start training without any pretrained model
2. **Resume from specific checkpoint**: Continue training from a specific checkpoint
3. **Resume from latest checkpoint**: Automatically find and resume from the latest checkpoint

### Command Line Options

```bash
# Train from scratch
python Main.py --is_training=True --is_pretrain=False --resume_from_latest=False

# Resume from specific checkpoint
python Main.py --is_training=True --is_pretrain=True --global_step=562800 --resume_from_latest=False

# Resume from latest checkpoint
python Main.py --is_training=True --is_pretrain=False --resume_from_latest=True
```

### Shell Scripts

We provide convenient shell scripts for training:

- `train.sh`: Standard training with memory optimization
- `train_mem_optimized.sh`: Further memory-optimized training for limited GPU memory

## Usage

### Training

```bash
# Make the script executable (Linux/Mac)
chmod +x train.sh

# Run training
./train.sh

# Or on Windows
bash train.sh
```

### Testing

```bash
python Main.py --is_training=False --is_testing=True
```

### Prediction

```bash
python Main.py --is_training=False --is_predicting=True
```

## Configuration

The `config.py` file contains all the configuration options. You can override any configuration parameter through command line arguments:

```bash
python Main.py --is_training=True --batch_size=32 --max_epoch=20
```

## Model Checkpoints

Model checkpoints are saved in the `Checkpoints` directory, organized by instrument and model type. The system automatically finds the latest checkpoint when using the `resume_from_latest=True` option.

## Data Structure

- `NoiseDataforHoribaLabRAM`: Training data
- `Testingdataset`: Testing data
- `Predict`: Data for prediction
- `Model`: Pretrained models

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- TensorBoard