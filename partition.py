from typing import List, Tuple
from collections import Counter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import numpy as np
import random
import re
import os
import json

LABELS_FOLDER_NUM = {"E": 8, "I": 4, "O": 4}
LABELS_DIRS_INDEX = {"E": [], "I": [], "O": []}

VALID_EXTENSIONS = {".png", ".jpg"}

FAST_DEBUG = False

LOAD_INDICES = True


# Function to create a validator function with additional arguments
def create_image_file_validator(valid_extensions):
    def is_valid_image_file(filepath):
        return any(filepath.lower().endswith(ext) for ext in valid_extensions)

    return is_valid_image_file


# Specify valid extensions
valid_extensions = [".jpg", ".jpeg", ".png"]
# Create the validator function
is_valid_image_file = create_image_file_validator(valid_extensions)


# Version 1
# Function to balance the dataset
def balance_dataset(dataset):
    targets = dataset.targets
    class_indices = {target: [] for target in set(targets)}
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)

    min_class_size = min(len(indices) for indices in class_indices.values())

    balanced_indices = []
    for indices in class_indices.values():
        balanced_indices.extend(random.sample(indices, min_class_size))

    return balanced_indices


def get_datasets_indices_ver1(dataset, train_portion, val_portion):
    balanced_indices = balance_dataset(dataset)

    # Randomly shuffle the indices
    random.shuffle(balanced_indices)

    # Calculate the sizes of each split
    total_size = len(balanced_indices)
    train_size = int(total_size * train_portion)
    val_size = int(total_size * val_portion)

    # Split the indices
    train_indices = balanced_indices[:train_size]
    val_indices = balanced_indices[train_size : train_size + val_size]
    test_indices = balanced_indices[train_size + val_size :]

    return train_indices, val_indices, test_indices


# Version 2
def valid_extension(path):
    _, ext = os.path.splitext(path)
    return ext.lower() in VALID_EXTENSIONS


def valid_train_idx(sub_folder_idx, class_label):
    return sub_folder_idx in {
        LABELS_DIRS_INDEX[class_label][0],
        LABELS_DIRS_INDEX[class_label][1],
    }


def valid_val_idx(sub_folder_idx, class_label):
    return sub_folder_idx in {LABELS_DIRS_INDEX[class_label][2]}


def valid_test_idx(sub_folder_idx, class_label):
    return sub_folder_idx in {LABELS_DIRS_INDEX[class_label][3]}


def get_datasets_indices_ver2(dataset, train_portion, val_portion):

    for label in LABELS_FOLDER_NUM.keys():
        arr = np.arange(1, LABELS_FOLDER_NUM[label] + 1)
        np.random.shuffle(arr)

        LABELS_DIRS_INDEX[label] = arr

        print(f"Label: {label}, \t{LABELS_DIRS_INDEX[label]}")

    # Split the indices
    train_indices = []
    val_indices = []
    test_indices = []

    for idx, imgs in enumerate(dataset.imgs):
        path = imgs[0]
        label = imgs[1]

        if not valid_extension(path):
            continue

        sub_folder_idx = int(re.split(r"[\\/]", path)[-2][1:])
        class_label = dataset.classes[label]

        if valid_train_idx(sub_folder_idx, class_label):
            train_indices.append(idx)
        elif valid_val_idx(sub_folder_idx, class_label):
            val_indices.append(idx)
        elif valid_test_idx(sub_folder_idx, class_label):
            test_indices.append(idx)

    return train_indices, val_indices, test_indices


def targets_in_dataset_and_subset(dataset):
    if isinstance(dataset, ImageFolder):
        return dataset.targets
    elif isinstance(dataset, Subset):
        ori_targets = np.array(targets_in_dataset_and_subset(dataset.dataset))
        return ori_targets[dataset.indices]


def indices_in_dataset_and_subset(dataset):
    if isinstance(dataset, ImageFolder):
        return list(range(len(dataset)))
    elif isinstance(dataset, Subset):
        ori_indices = np.array(indices_in_dataset_and_subset(dataset.dataset))
        return ori_indices[dataset.indices]


def get_dataloader(
    base_folder: str,
    ver: int,
    train_portion: float,
    val_portion: float,
    batch_size: int,
    transform: transforms.Compose,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dataset = ImageFolder(root=base_folder, transform=transform)

    # Save indices as JSON files
    train_indices_path = "./models/train_indices_ver_" + str(ver) + ".json"
    val_indices_path = "./models/val_indices_ver_" + str(ver) + ".json"
    test_indices_path = "./models/test_indices_ver_" + str(ver) + ".json"

    if LOAD_INDICES and os.path.exists(train_indices_path) and os.path.exists(val_indices_path) and os.path.exists(test_indices_path):
        with open(train_indices_path, "r") as f:
            train_indices = json.load(f)
        with open(val_indices_path, "r") as f:
            val_indices = json.load(f)
        with open(test_indices_path, "r") as f:
            test_indices = json.load(f)
    else:
        train_indices, val_indices, test_indices = (
            get_datasets_indices_ver1(dataset, train_portion, val_portion)
            if ver == 1
            else get_datasets_indices_ver2(dataset, train_portion, val_portion)
        )

        with open(train_indices_path, "w") as f:
            json.dump(train_indices, f)
        with open(val_indices_path, "w") as f:
            json.dump(val_indices, f)
        with open(test_indices_path, "w") as f:
            json.dump(test_indices, f)

    if FAST_DEBUG:
        train_indices = train_indices[:10]
        val_indices = val_indices[:10]
        test_indices = test_indices[:10]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Calculate the number of examples for each class
    print(f"Dataset Split:")
    print(
        f"Training size: {len(train_dataset)}, {Counter(targets_in_dataset_and_subset(train_dataset))}"
    )
    print(
        f"Validation size: {len(val_dataset)}, {Counter(targets_in_dataset_and_subset(val_dataset))}"
    )
    print(
        f"Testing size: {len(test_dataset)}, {Counter(targets_in_dataset_and_subset(test_dataset))}"
    )

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
