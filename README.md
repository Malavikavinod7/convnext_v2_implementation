# ConvNeXt V2 Image Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A clean and practical PyTorch implementation for image classification using ConvNeXt-based backbones. This project is designed to be easy to run, inspect, and share on GitHub.

## Overview

This project trains a ConvNeXt-based image classifier on a folder-based dataset structure and provides utilities for:

- splitting a dataset into train/validation folders
- training the model
- evaluating predictions and saving reports/plots
- loading and inspecting the best checkpoint

## Features

- ConvNeXt Tiny and Base backbone support
- Folder-based dataset loading with PyTorch ImageFolder
- Mixed-precision training support
- CSV logging for training metrics
- Automatic generation of evaluation plots and reports
- Portable CLI-based scripts for training and inference

## Project Structure

- [train.py](train.py) — training pipeline
- [model.py](model.py) — model builder
- [dataset.py](dataset.py) — dataset loading helpers
- [Split_dataset.py](Split_dataset.py) — dataset split utility
- [plot_and_predict.py](plot_and_predict.py) — evaluation and visualization
- [check.py](check.py) — checkpoint inspection utility

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Prepare your dataset such that the root contains one folder per class:

```text
data/
  class_1/
    img1.jpg
    img2.jpg
  class_2/
    img1.jpg
```

Then split it into train/validation data:

```bash
python Split_dataset.py --data-dir data --output-dir data_split
```

## Training

```bash
python train.py --data-dir data_split --output-dir outputs --epochs 20 --batch-size 16
```

## Evaluation

```bash
python plot_and_predict.py --data-dir data_split/val --output-dir outputs
```

## Notes

- The project expects a folder structure compatible with PyTorch ImageFolder.
- A checkpoint named best_model.pth will be written to the output directory after training.
- You can customize the backbone with the model_name argument if needed.
- The workflow in [.github/workflows/train.yml](.github/workflows/train.yml) provides a simple CI example for automated training.

## License

This project is available under the MIT License.
