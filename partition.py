from typing import List, Tuple
from collections import Counter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import numpy as np
import random
import re
import os

LABELS_FOLDER_NUM = {"E": 8, "I": 4, "O": 4}
LABELS_DIRS_INDEX = {"E": [], "I": [], "O": []}

FAST_DEBUG = False


# Version 1
# Function to balance the dataset
def balance_dataset(dataset):
    targets = dataset.targets
    class_indices = {target: [] for target in set(targets)}
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)

    min_class_size = min(len(indices) for indices in class_indices.values())
    min_class_size = 10 if FAST_DEBUG else min_class_size

    balanced_indices = []
    for indices in class_indices.values():
        balanced_indices.extend(random.sample(indices, min_class_size))

    return balanced_indices


def get_datasets_ver1(base_folder, train_portion, val_portion, transform):
    dataset = ImageFolder(root=base_folder, transform=transform)

    # Get balanced indices
    balanced_indices = balance_dataset(dataset)

    # Create a balanced subset of the dataset
    balanced_dataset = Subset(dataset, balanced_indices)
    # print(
    #     f"Balanced dataset size: {len(balanced_dataset)}, \tClass distribution: {Counter(balanced_dataset.targets)}"
    # )

    # Randomly split the balanced dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        balanced_dataset,
        [train_portion, val_portion, 1.0 - train_portion - val_portion],
    )

    return train_dataset, val_dataset, test_dataset


# Version 2
def is_valid_file_train(path):
    _, ext = os.path.splitext(path)
    is_valid_extension = ext.lower() in {".png", ".jpg"}
    if not is_valid_extension:
        return False

    index = int(re.split(r"[\\/]", path)[-2][1:])
    label = re.split(r"[\\/]", path)[-2][0]

    return index in {LABELS_DIRS_INDEX[label][0], LABELS_DIRS_INDEX[label][1]}


def is_valid_file_val(path):
    _, ext = os.path.splitext(path)
    is_valid_extension = ext.lower() in {".png", ".jpg"}
    if not is_valid_extension:
        return False

    index = int(re.split(r"[\\/]", path)[-2][1:])
    label = re.split(r"[\\/]", path)[-2][0]

    return index in {LABELS_DIRS_INDEX[label][2]}


def is_valid_file_test(path):
    _, ext = os.path.splitext(path)
    is_valid_extension = ext.lower() in {".png", ".jpg"}
    if not is_valid_extension:
        return False

    index = int(re.split(r"[\\/]", path)[-2][1:])
    label = re.split(r"[\\/]", path)[-2][0]

    return index in {LABELS_DIRS_INDEX[label][3]}


# Function to load datasets from directories
def load_datasets(base_folder, transform, is_valid_file):
    datasets = ImageFolder(
        root=base_folder, transform=transform, is_valid_file=is_valid_file
    )

    datasets = Subset(datasets, indices=range(10)) if FAST_DEBUG else datasets

    return datasets


def get_datasets_ver2(base_folder, transform):
    for label in LABELS_FOLDER_NUM.keys():
        arr = np.arange(1, LABELS_FOLDER_NUM[label] + 1)
        np.random.shuffle(arr)

        LABELS_DIRS_INDEX[label] = arr

        print(f"Label: {label}, \t{LABELS_DIRS_INDEX[label]}")

    # Load datasets for each phase
    train_dataset = load_datasets(
        base_folder=base_folder, transform=transform, is_valid_file=is_valid_file_train
    )
    val_dataset = load_datasets(
        base_folder=base_folder, transform=transform, is_valid_file=is_valid_file_val
    )
    test_dataset = load_datasets(
        base_folder=base_folder, transform=transform, is_valid_file=is_valid_file_test
    )

    return train_dataset, val_dataset, test_dataset


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
    # Create ImageFolder dataset
    train_dataset, val_dataset, test_dataset = (
        get_datasets_ver1(
            base_folder=base_folder,
            train_portion=train_portion,
            val_portion=val_portion,
            transform=transform,
        )
        if ver == 1
        else get_datasets_ver2(base_folder=base_folder, transform=transform)
    )

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
