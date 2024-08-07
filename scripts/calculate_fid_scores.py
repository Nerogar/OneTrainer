import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import Dataset
from PIL import Image
import json

# Set up the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations for the images
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to the required input size of Inception v3
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

# Custom loader function
def loader(path):
    return Image.open(path).convert('RGB')

class EpochDataset(Dataset):
    def __init__(self, epoch_path, transform=None, loader=None):
        self.epoch_path = epoch_path
        self.transform = transform
        self.loader = loader
        self.image_files = [f for f in os.listdir(epoch_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.epoch_path, image_file)
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_files)

def calculate_fid_scores(validation_images_path, epochs_path):
    # Load the validation images using ImageFolder with the custom loader
    validation_dataset = ImageFolder(validation_images_path, transform=transform, loader=loader)

    # Create an instance of the FrechetInceptionDistance metric
    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Load existing FID scores if the file exists
    fid_scores_file = os.path.join(epochs_path, "fid_scores.json")
    if os.path.exists(fid_scores_file):
        with open(fid_scores_file, "r") as f:
            epoch_fid_scores = json.load(f)
    else:
        epoch_fid_scores = {}

    # Get the list of epoch folders sorted in ascending order
    epoch_folders = sorted([folder for folder in os.listdir(epochs_path) if folder.startswith("class_")])

    # Get the latest epoch folder
    latest_epoch_folder = epoch_folders[-1]

    # Extract the epoch number from the latest epoch folder name
    latest_epoch_number = int(latest_epoch_folder.split("_")[-1])

    # Calculate FID score only for the latest epoch
    epoch_path = os.path.join(epochs_path, latest_epoch_folder)
    # Load the generated images for the latest epoch using the custom dataset
    generated_dataset = EpochDataset(epoch_path, transform=transform, loader=loader)

    # Check if both validation and generated datasets have at least two samples
    if len(validation_dataset) >= 2 and len(generated_dataset) >= 2:
        # Calculate the FID score for the latest epoch
        fid.reset()
        fid.update(torch.stack([img.to(device) for img, _ in validation_dataset]), real=True)
        fid.update(torch.stack([img.to(device) for img in generated_dataset]), real=False)
        fid_score = fid.compute()

        # Store the FID score for the latest epoch using the epoch number as the key
        epoch_fid_scores[str(latest_epoch_number)] = fid_score.item()
    else:
        print(f"Skipping FID calculation for epoch {latest_epoch_folder} due to insufficient samples.")

    # Print the FID scores for each epoch
    for epoch, fid_score in epoch_fid_scores.items():
        print(f"Epoch {epoch}: FID score = {fid_score}")

    # Store updated FID scores in the JSON file
    with open(fid_scores_file, "w") as f:
        json.dump(epoch_fid_scores, f)
        
    # Return the epoch_fid_scores dictionary
    return epoch_fid_scores
