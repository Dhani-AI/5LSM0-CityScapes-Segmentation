# 5LSM0-CityScapes-Segmentation

- **Author:** Dhani Crapels
- **Codalab Username:** Dhani
- **Codalab Competition:** [5LSM0 competition (2024)](https://codalab.lisn.upsaclay.fr/competitions/17868#learn_the_details)
- **TU/e Email Address:** d.r.m.crapels@student.tue.nl

## Overview
This assignment is part of the 5LSM0 course taught at Eindhoven University of Technology (TU/e) in 2023-2024. It involves working with the Cityscapes dataset and training a neural network. This repository contains the code for semantic segmentation of cityscapes using a U-net model. The implementation is based on the [starting kit for the 5LSM0 final assignment](https://github.com/5LSM0/FinalAssignment) 

## Getting Started

### Installing
To get started with this project, you need to clone the repository to your local machine. You can do this by running the following command in your terminal:
```bash
git clone https:https://github.com/Dhani-AI/5LSM0-CityScapes-Segmentation.git
```

After cloning the repository, navigate to the project directory:
```bash
cd 5LSM0-CityScapes-Segmentation
```

To run the code, you will need the following libraries:

- PyTorch
- NumPy
- Matplotlib
- torchmetrics
- torchvision
- wandb

To install the required libraries, simply run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Dataset

The code expects the Cityscapes dataset to be downloaded. You can obtain the dataset from the [official Cityscapes website](https://www.cityscapes-dataset.com/). Once downloaded, make sure to organize the dataset as follows:

1. Create a root directory named `dataset`.
2. Inside the `dataset` directory, create two subdirectories: `leftImg8bit` and `gtFine`.
3. Place the Cityscapes images (RGB images) in the `leftImg8bit` directory.
4. Place the corresponding ground truth annotations (semantic segmentation masks) in the `gtFine` directory.

The directory structure should look like this:

```plaintext
dataset/
├── leftImg8bit/
│   ├── city1/
│   ├── city2/
│   └── ...
└── gtFine/
    ├── city1/
    ├── city2/
    └── ...
```

## File Descriptions

- `train.py`: Script for training the segmentation model.
- `model.py`: Script that defines the model architecture.
- `utils.py`: Utility functions designed to facilitate the accurate mapping between the 19 training classes
- `process_data.py`: Script containing functions for preprocessing and postprocessing data.
- `run_container.sh`: (Optional) Contains the script for running the container. This file may include settings such as wandb keys and additional arguments, but it is not essential for training the model.
- `run_main`: (Optional) Includes the code for building the Docker container. This file is not directly related to training the model.

## Usage

### Training the Model

To train the segmentation model, you can use the `train.py` script. Make sure you have the Cityscapes dataset downloaded and preprocessed as per the instructions provided in the dataset section. You can start training by running the following command:

```bash
python train.py
```

### Experiment Tracking with Weights & Biases (wandb)

Weights & Biases (wandb) is used to track experiments and monitor model performance. Before running the training script, make sure to set up a wandb account and initialize your API key when prompted, using the following command:

```bash
wandb login
```

Also see the website for more information on how to setup [Weights & Biases](https://docs.wandb.ai/quickstart)
