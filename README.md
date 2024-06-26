# CycleGAN Trajectory reconstruction

## Project Overview
This project aims to implement a CycleGAN for image denoising and trajectory reconstruction using a custom dataset of trajectory images generated by a soft pneumatic robotic arm. The project is organized into several modules with configurable settings for easy experimentation and extension.

## File Structure
- `config/`
  - `config.json`: Configuration file for setting model parameters, training settings, and noise types.
- `models/`
  - `generators.py`: Defines the `ResnetGenerator` class with ResNet blocks.
  - `discriminators.py`: Defines the `NLayerDiscriminator` class.
  - `losses.py`: Defines the loss functions used for training.
- `utils/`
  - `dataset.py`:  Contains `ImageDataLoader` for loading and preprocessing the dataset.
  - `noise.py`: Contains `NoiseAdder` for adding noise to the images.
  - `plot.py`: Functions for plotting images.
  - `stats.py`: Functions for printing training statistics.
  - `validate.py`: Script for validating the model during training.
  - `test.py`: Script for testing the model after training.
- `train.py`: Main script for training the models.
- `main.py`: Entry point for starting the training or testing based on the configuration.

## Configuration Guide

### General Settings
- `show_plots`: Whether to display plots.
- `save_plots`: Whether to save plots.
- `mode`: Mode of operation (`train` or `test`).
- `dataset_path`: Path to the dataset.

### Preprocessing Settings
- `normalization`: Apply normalization to the images.
- `standardization`: Apply standardization to the images.
- `data_augmentation`: Apply data augmentation techniques.

### Noise Adder Settings
- `noise_type`: Type of noise to add (`line_segments`, `salt_and_pepper`, `gaussian`, etc.).
- `salt_pepper_ratio`: Ratio of salt and pepper noise.
- `gaussian_mean`: Mean of Gaussian noise.
- `gaussian_std`: Standard deviation of Gaussian noise.
- `line_segment_params`: Parameters for line segment noise (number of lines and thickness).

### Training Settings
- `training_size`: Number of training samples.
- `validation_size`: Number of validation samples.
- `test_size`: Number of test samples.
- `seed`: Random seed for reproducibility.
- `batch_size`: Number of samples per gradient update.
- `num_epochs`: Number of epochs to train the model.
- `generator_learning_rate`: Learning rate for the generator.
- `discriminator_learning_rate`: Learning rate for the discriminator.
- `beta1`, `beta2`: Coefficients for the Adam optimizer.
- `lambda_cycle`: Weight for cycle consistency loss.
- `lambda_identity`: Weight for identity loss.
- `loss_function`: Loss function to use (`BCE`, `MSE`).
- `optimizer`: Optimizer to use (`Adam`).
- `gradient_clipping`: Whether to apply gradient clipping.

### Generator Settings
- **Input Dimension**: [128, 128, 1]
- **Layers**:
  1. **Conv Layer**: Out Channels: 64, Kernel Size: 7, Stride: 1, Activation: ReLU, Batch Norm: Yes, Dropout: 0.3
  2. **Conv Layer**: Out Channels: 128, Kernel Size: 3, Stride: 2, Activation: ReLU, Batch Norm: Yes, Dropout: 0.3
  3. **Conv Layer**: Out Channels: 256, Kernel Size: 3, Stride: 2, Activation: ReLU, Batch Norm: Yes, Dropout: 0.3
  4. **Residual Blocks (6)**: Out Channels: 256, Kernel Size: 3, Activation: ReLU, Batch Norm: Yes, Dropout: 0.3
  5. **Conv Transpose Layer**: Out Channels: 128, Kernel Size: 3, Stride: 2, Activation: ReLU, Batch Norm: Yes, Dropout: 0.3
  6. **Conv Transpose Layer**: Out Channels: 64, Kernel Size: 3, Stride: 2, Activation: ReLU, Batch Norm: Yes, Dropout: 0.3
  7. **Conv Layer**: Out Channels: 1, Kernel Size: 7, Stride: 1, Activation: Tanh, Batch Norm: No

### Discriminator Settings
- **Input Dimension**: [128, 128, 1]
- **Layers**:
  1. **Conv Layer**: Out Channels: 32, Kernel Size: 4, Stride: 2, Activation: LeakyReLU, Batch Norm: Yes, Dropout: 0.3
  2. **Conv Layer**: Out Channels: 64, Kernel Size: 4, Stride: 2, Activation: LeakyReLU, Batch Norm: Yes, Dropout: 0.3
  3. **Conv Layer**: Out Channels: 128, Kernel Size: 4, Stride: 2, Activation: LeakyReLU, Batch Norm: Yes, Dropout: 0.3
  4. **Conv Layer**: Out Channels: 1, Kernel Size: 4, Stride: 1, Activation: Sigmoid, Batch Norm: No

## Running the Project

### Python Version
Ensure you are using Python 3.9 to be compatible with the version of TensorFlow used in this project, which allows for GPU utilization. For detailed instructions on setting up TensorFlow with GPU support, visit: [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip?hl=it#windows-native_1)

1. Install the required dependencies:
   ```bash
   pip install -r requirements
   ```
2. Run the training script:
   ```bash
   python main.py
   ```

3. Monitor the output and adjust configurations as needed.