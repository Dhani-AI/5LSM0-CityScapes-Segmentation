"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import numpy as np
import os
import utils
import process_data
from torchmetrics.classification import MulticlassJaccardIndex, Dice
from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import wandb

IMAGE_HEIGHT = 256  # Original image height 1024
IMAGE_WIDTH = 512  # Original image width 2048
NUM_EPOCHS = 25
LEARNING_RATE = 0.0004
BATCH_SIZE = 32
IGNORE_INDEX = 255
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
NUM_CLASSES = 19

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


def main(args):
    """Main training loop for the CityScapes segmentation model"""
 
    # start a new wandb run
    wandb.init(
        project="CityScapes-Segmentation",  # change this to your project name
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "UNet",
            "dataset": "CityScapes",
            "epochs": NUM_EPOCHS,
        }
    )

    # Define transforms for both the image and its annotation
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)
    ])

    annotation_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor()
    ])

    # data loading
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Accessing an example from the training and validation dataset
    train_img, train_semantic = train_dataset[0]
    val_img, val_semantic = val_dataset[0]

    # Visualize the examples        
    visualize_sample(train_img, train_semantic, CITYSCAPES_MEAN, CITYSCAPES_STD, 'train_example.png')
    visualize_sample(val_img, val_semantic, CITYSCAPES_MEAN, CITYSCAPES_STD, 'val_example.png')

    # Model setup
    model = Model().to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function (ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Training / Validation loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        train_jaccard = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).to(DEVICE)
        train_jaccard.reset()

        for images, target in train_loader:
            images = images.to(DEVICE)
            target = (target*255).long().squeeze()  # because the id are normalized between 0-1
            target = utils.map_id_to_train_id(target).to(DEVICE)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, target)  # compute loss

            # Backward pass
            optimizer.zero_grad()  # set the gradients to zero
            loss.backward()
            optimizer.step()  # update the weights

            # Compute the loss
            train_loss += loss.item()
            
            # Postprocess predictions to generate masks
            masks_pred = process_data.postprocess(predictions, (IMAGE_HEIGHT, IMAGE_WIDTH))
            
            # Convert NumPy array to PyTorch tensor
            masks_pred_tensor = torch.tensor(masks_pred, dtype=torch.long).to(DEVICE)
            train_jaccard.update(masks_pred_tensor.flatten(), target.flatten())

        # Calculate average training loss and dice coefficient
        train_loss = train_loss / len(train_loader)
        train_jaccard_score = train_jaccard.compute()

        # Evaluation on the validation set
        model.eval()
        val_loss = 0.0
        val_jaccard = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255).to(DEVICE)
        val_jaccard.reset()

        with torch.inference_mode():
            for images_val, target_val in val_loader:
                images_val = images_val.to(DEVICE)
                target_val = (target_val * 255).long().squeeze()  # Because IDs are normalized between 0-1
                target_val = utils.map_id_to_train_id(target_val).to(DEVICE)

                # Forward pass
                predictions_val = model(images_val)
                loss = criterion(predictions_val, target_val)

                # Compute the loss
                val_loss += loss.item()
                
                # Postprocess predictions to generate masks
                masks_pred_val = process_data.postprocess(predictions_val, (IMAGE_HEIGHT, IMAGE_WIDTH))
  
                # Convert NumPy array to PyTorch tensor
                masks_pred_tensor_val = torch.tensor(masks_pred_val, dtype=torch.long).to(DEVICE)
                val_jaccard.update(masks_pred_tensor_val.flatten(), target_val.flatten())

        # Calculate average validation loss
        val_loss = val_loss / len(val_loader)
        val_jaccard_score = val_jaccard.compute()

        # Print the training and validation loss for each epoch
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "Train_Jaccard": train_jaccard_score,
            "Val_Jaccard": val_jaccard_score,
        })

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
            annotation = (annotation * 255).long().squeeze()  # Because IDs are normalized between 0-1
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
