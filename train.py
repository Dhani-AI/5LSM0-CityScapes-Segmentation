"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
# Import python files
import utils
import process_data
from model import Model

# Import necessary libraries
import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torchmetrics.functional.classification import (accuracy, dice,
                                                    multiclass_jaccard_index)

from torchvision.datasets import Cityscapes
from torchvision import transforms
from torchvision.transforms import InterpolationMode, v2
from torch.utils.data import DataLoader, random_split

IMAGE_HEIGHT = 256  # Original image height 1024
IMAGE_WIDTH = 512  # Original image width 2048
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 2
SCHEDULER_STEPSIZE = 10
SCHEDULER_GAMMA = 0.1
WEIGHT_DECAY = 0.0001
IGNORE_INDEX = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
NUM_CLASSES = 20

CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def visualize_sample(normalized_image, annotation, mean, std, filename):
    """Visualize the image and its annotation"""
    
    # Inverse normalize the image
    inverse_normalize = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
    ])
    original_image = inverse_normalize(normalized_image)
    
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image.permute(1, 2, 0))  # Convert tensor back to PIL Image for visualization
    
    # Plot normalized image
    plt.subplot(1, 3, 2)
    plt.title('Normalized Image')
    plt.imshow(normalized_image.permute(1, 2, 0))  # Convert tensor back to PIL Image for visualization
    
    # Plot Ground Truth
    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    annotation = annotation.squeeze().numpy()
    plt.imshow(annotation, cmap='jet')  # The annotation is a tensor
    
    # Save the visualization
    plt.savefig(filename)
    plt.close()
    
    
def visualize_result(original_image, annotation, predicted_annotation, filename):
    """Visualize a single original image, ground truth annotation, and predicted annotation."""
    # Reverse normalization of the original image
    original_image = original_image.cpu().numpy()
    original_image = np.transpose(original_image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    original_image = original_image * CITYSCAPES_MEAN + CITYSCAPES_STD
        
    # Plot the images
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    # Ground truth annotation
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth')
    plt.imshow(annotation.squeeze().cpu().numpy(), vmin = 0, vmax = 18, cmap='jet')
    plt.axis('off')

    # Predicted annotation
    plt.subplot(1, 3, 3)
    plt.title('Predicted Segmentation')
    plt.imshow(predicted_annotation.squeeze().cpu().numpy(), vmin = 0, vmax = 18, cmap='jet')
    plt.axis('off')

    # Save the visualization
    plt.savefig(filename)
    plt.close()


def create_data_loaders(args, batch_size, image_height, image_width, mean, std, num_workers):
    """Create the data loaders for the CityScapes dataset"""
    # Define transforms for both the image and its annotation
    transform = v2.Compose([
        v2.Resize((image_height, image_width)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    annotation_transform = v2.Compose([
        v2.Resize((image_height, image_width), interpolation=InterpolationMode.NEAREST),
        v2.ToImage()
    ])

    # Load the dataset
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform,
                         target_transform=annotation_transform)
    
    # dataset[0]     //pair image mask
    # dataset[0][0]  //image  dim [3, 1024, 2048] (channels, height, width) type  torch.Tensor
    # dataset[0][1]  //mask   dim [1, 1024, 2048] (channels, height, width) type  torch.Tensor

    # Define the size of the validation set and the training set
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, val_dataset, train_loader, val_loader


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    train_iou = 0.0

    for images, targets in tqdm(train_loader):
        images = images.to(device)
        targets = targets.long().squeeze(dim=1)
        targets = utils.map_id_to_train_id(targets)
        targets[targets == 255] = 19
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach()
        outputs_max = torch.argmax(predictions, dim=1)
        
        train_acc += accuracy(outputs_max, targets, task="multiclass", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).detach()
        train_dice += dice(predictions, targets, ignore_index=IGNORE_INDEX).detach()
        train_iou += multiclass_jaccard_index(predictions, targets, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).detach()
    
    num_batches = len(train_loader)
    return (train_loss / num_batches).item(), (train_acc / num_batches).item(), (train_dice / num_batches).item(), (train_iou / num_batches).item()


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            targets = (targets).long().squeeze(dim=1) 
            targets = utils.map_id_to_train_id(targets)
            targets[targets == 255] = 19
            targets = targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.detach()
            outputs_max = torch.argmax(predictions, dim=1)
            
            val_acc += accuracy(outputs_max, targets, task="multiclass", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).detach()
            val_dice += dice(predictions, targets, ignore_index=IGNORE_INDEX).detach()
            val_iou += multiclass_jaccard_index(predictions, targets, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).detach()
        
    num_batches = len(val_loader)
    return (val_loss / num_batches).item(), (val_acc / num_batches).item(), (val_dice / num_batches).item(), (val_iou / num_batches).item()

def main(args):
    """Main training loop for the CityScapes segmentation model"""
 
    # start a new wandb run
    wandb.init(
        project="CityScapes-Segmentation",  # change this to your project name
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "num_workers": NUM_WORKERS,
            "architecture": "UNet",
            "dataset": "CityScapes",
            "optimizer": "Adam",

        }
    )

    # Get the data loaders
    train_dataset, val_dataset, train_loader, val_loader = create_data_loaders(args, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CITYSCAPES_MEAN, CITYSCAPES_STD, NUM_WORKERS)

    # Accessing an example from the training and validation dataset
    train_img, train_semantic = train_dataset[0]
    val_img, val_semantic = val_dataset[0]

    # Visualize the examples        
    visualize_sample(train_img, train_semantic, CITYSCAPES_MEAN, CITYSCAPES_STD, 'train_example.png')
    visualize_sample(val_img, val_semantic, CITYSCAPES_MEAN, CITYSCAPES_STD, 'val_example.png')

    # Model setup
    model = Model().to(DEVICE)

    # Loss function (ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEPSIZE, gamma=SCHEDULER_GAMMA)

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_dices = []
    val_dices = []
    train_ious = []
    val_ious = []

    # Training / Validation loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc, train_dice, train_iou = train_model(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_dice, val_iou = validate_model(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_dice": train_dice,
            "train_lr": optimizer.param_groups[0]["lr"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_dice": val_dice,
            "val_lr": optimizer.param_groups[0]["lr"]
        })

        scheduler.step()

    wandb.finish()

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    # Visualize the results
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    # Define the folder path to save the visualizations
    result_folder = "results"
    os.makedirs(result_folder, exist_ok=True)

    with torch.inference_mode():
        for i, (image, annotation) in enumerate(val_loader):
              
            # Move data to device
            image = image.to(DEVICE)
            annotation = (annotation).long().squeeze(dim=1)  # Because IDs are normalized between 0-1
            annotation = utils.map_id_to_train_id(annotation).to(DEVICE)
            
            # Forward pass
            predicted_annotation = model(image)
            
            # Postprocess predictions
            masks_pred = process_data.postprocess(predicted_annotation, (IMAGE_HEIGHT, IMAGE_WIDTH))
            masks_pred_tensor = torch.tensor(masks_pred, dtype=torch.float32).to(DEVICE)
        
            # Visualize each sample in the batch
            for j in range(len(image)):
              visualize_result(image[j], annotation[j], masks_pred_tensor[j], os.path.join(result_folder, f'result_batch_{i}_sample_{j}.png'))
        
            break
    

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
