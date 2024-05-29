import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import re

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset class that inherits from ImageFolder
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, is_valid_file=None):
        super(CustomImageFolder, self).__init__(root, transform, is_valid_file)

    def __getitem__(self, index):
        # Get the original tuple from ImageFolder
        path, target = self.samples[index]
        # Load the image
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # Extract the index from the file path
        idx_match = re.search(r'_(\d+)\.png$', path)
        idx = int(idx_match.group(1)) if idx_match else -1
        return idx, sample, target

# Specify valid extensions
valid_extensions = ['.jpg', '.jpeg', '.png']

# Function to create a validator function with additional arguments
def create_image_file_validator(valid_extensions):
    def is_valid_image_file(filepath):
        return any(filepath.lower().endswith(ext) for ext in valid_extensions)
    return is_valid_image_file

# Create the validator function
is_valid_image_file = create_image_file_validator(valid_extensions)

# Specify directories for training, validation, and testing datasets
train_dir = "../_DatasetsLocal/CompoundEyeClassification/TrainData"
val_dir = "../_DatasetsLocal/CompoundEyeClassification/ValData"
test_dir = "../_DatasetsLocal/CompoundEyeClassification/TestData"

# Create ImageFolder datasets with the is_valid_file parameter
train_dataset = CustomImageFolder(root=train_dir, transform=transform, is_valid_file=is_valid_image_file)
val_dataset = CustomImageFolder(root=val_dir, transform=transform, is_valid_file=is_valid_image_file)
test_dataset = CustomImageFolder(root=test_dir, transform=transform, is_valid_file=is_valid_image_file)

# Print the sizes of the datasets
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")

# Define batch size
BATCH_SIZE = 32

# Create data loaders for each phase
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Iterate over the test loader and print the idx, inputs, and labels
for idx, inputs, labels in test_loader:
    print(f"Index: {idx}, Inputs: {inputs.shape}, Labels: {labels}")
