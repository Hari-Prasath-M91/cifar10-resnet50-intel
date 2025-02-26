# CIFAR-10 Image Classification with ResNet50

This repository provides a PyTorch implementation of a CIFAR-10 image classifier using the ResNet50 architecture, with optional Intel GPU (XPU) support activated via [Intel GPU Support for PyTorch 2.5](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/).

## Project Overview

The CIFAR-10 dataset consists of 60,000 32×32 color images in 10 classes. In this project, we use a modified ResNet50 model (pretrained on ImageNet) to classify CIFAR-10 images. Key features include:

- **Data Augmentation**: Resizing, random flips, rotations, color jitter
- **Mixed Precision Training**: Using `torch.amp.autocast`
- **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Incremental Training**: Each run trains for one epoch and saves the model state to `checkpoint.pth`, so the next run continues from the following epoch
- **Intel XPU Support**: Accelerated training on Intel GPUs

> **Note**: By default, ResNet50 outputs 1000 classes (ImageNet). To adapt it for CIFAR-10, change the final layer:
> ```python
> model.fc = torch.nn.Linear(model.fc.in_features, 10)
> ```

## Table of Contents

- [Getting Started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Data Visualization](#data-visualization)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Performance Metrics](#performance-metrics)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### Install Dependencies

All required packages (including PyTorch 2.5 with Intel GPU support) are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

> **Clarification**: To run on Intel GPUs (XPU), this project relies on [Intel GPU Support for PyTorch 2.5](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/). Make sure your system meets the hardware and driver requirements detailed in that blog post. If you have a compatible Intel GPU and the correct PyTorch build, the scripts will detect and utilize the GPU automatically.

## Usage

### Data Visualization

To view a batch of CIFAR-10 images along with their labels, run:

```bash
python visualize_data.py
```

### Model Training

Before running the training script for the first time, **ensure that you modify the CIFAR-10 dataset initialization in your code** (typically in `model_training.py`) to set `download=True` so the dataset is downloaded. For example:

```python
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=True  # Set to True on the first run to download the dataset
)
```

After the dataset has been downloaded, you can change `download` to `False` to prevent re-downloading.

To train the ResNet50 model, run:

```bash
python model_training.py
```

Each run processes one epoch and saves the state to `checkpoint.pth`. Subsequent runs resume from the next epoch.

### Model Evaluation

To evaluate the model’s performance (accuracy, precision, recall, F1-score) and view a confusion matrix, run:

```bash
python accuracy_check.py
```

## Performance Metrics

After training the model for 20 epochs, the following performance metrics were obtained (using the checkpoint loaded from `checkpoint.pth`):

- **Accuracy:** 60.99%
- **Weighted Precision:** 60.61%
- **Weighted Recall:** 60.99%
- **Weighted F1-Score:** 60.55%

These metrics provide a baseline for the model's performance on CIFAR-10 with the current configuration.

## License

This project is licensed under the MIT License. See the `LICENSE` file in this repository for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
- [Intel GPU Support for PyTorch 2.5](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/)

## Contact

If you have questions, suggestions, or issues, please:

- Open an issue on GitHub  
- Feel free to contact me at [hariprasath.m2017@gmail.com](mailto:hariprasath.m2017@gmail.com)
