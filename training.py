import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from transformers import ResNetForImageClassification
from tqdm import tqdm
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# global variables
DATA_ROOT = "../_DatasetsLocal/CompoundEyeClassification/Data"
MODEL_PATH = "../_ModelsLocal/resnet-50"
SAVE_WEIGHTS = "./models/best_model.pth"
IMG_PATH = "./images/"
CONTINUE_TRAINING = True
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_CLASSES = 3
DENS_LEVEL = 10
TRAINING_PORTION = 0.8
VALIDATION_PORTION = 0.1
EPOCHS = 2

# debug settings
FAST_DEBUG = False

# 1. Create datasets & Data preprocessing
# Define data transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to match ResNet50 input size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize image pixels
    ]
)

# Create ImageFolder dataset
dataset = ImageFolder(root=DATA_ROOT, transform=transform)
print(
    f"Original dataset size: {len(dataset)}, \tClass distribution: {Counter(dataset.targets)}"
)


# Function to balance the dataset
def balance_dataset(dataset):
    targets = dataset.targets
    class_indices = {target: [] for target in set(targets)}
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)

    min_class_size = min(len(indices) for indices in class_indices.values())
    min_class_size = 100 if FAST_DEBUG else min_class_size

    balanced_indices = []
    for indices in class_indices.values():
        balanced_indices.extend(random.sample(indices, min_class_size))

    return balanced_indices


# Get balanced indices
balanced_indices = balance_dataset(dataset)

# Create a balanced subset of the dataset
balanced_dataset = Subset(dataset, balanced_indices)
# print(
#     f"Balanced dataset size: {len(balanced_dataset)}, \tClass distribution: {Counter(balanced_dataset.targets)}"
# )


# Define sizes for train, validation, and test sets
train_size = int(TRAINING_PORTION * len(balanced_dataset))
val_size = int(VALIDATION_PORTION * len(balanced_dataset))
test_size = len(balanced_dataset) - train_size - val_size

# Randomly split the balanced dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(
    balanced_dataset, [train_size, val_size, test_size]
)

print(f"Dataset Split:")
print(f"\tTraining size: {len(train_dataset)}")
print(f"\tValidation size: {len(val_dataset)}")
print(f"\tTesting size: {len(test_dataset)}")

# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 2. Load pre-trained model and adjust the classification head
model = ResNetForImageClassification.from_pretrained(
    MODEL_PATH, ignore_mismatched_sizes=True, num_labels=NUM_CLASSES
)

# Load the model weights
if CONTINUE_TRAINING and os.path.exists(SAVE_WEIGHTS):
    model.load_state_dict(torch.load(SAVE_WEIGHTS))


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Initialize lists to store training and validation loss
train_losses = []
val_losses = []
val_accuracies = []


# Updated Plotting function for loss and accuracy per epoch
def plot_epoch_loss(epoch, batch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, "b", label="Batch loss")
    plt.title(f"Training loss in Epoch {epoch+1}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(IMG_PATH + f"epoch_{epoch+1}_loss.png")  # Save plot as an image
    plt.close()


# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_losses = []  # List to store batch losses
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_losses.append(loss.item())  # Append batch loss

            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update the progress bar
            train_loop.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            train_loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)

        # Plot and save batch losses for the current epoch
        plot_epoch_loss(epoch, batch_losses)

        # Evaluate the model on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_loop = tqdm(val_loader, total=len(val_loader), leave=True)
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
                outputs = model(inputs)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs.logits, labels).item()

                # Update the progress bar
                val_loop.set_description(f"Validation Epoch {epoch + 1}/{num_epochs}")
                val_loop.set_postfix(val_loss=val_loss)

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), SAVE_WEIGHTS)


# Evaluation function with fog density analysis
def evaluate_model_with_fog_density(model, test_loader):
    # test accuracy plot grid
    dens_level = DENS_LEVEL

    model.eval()
    correct = 0
    total = 0
    density_correct = {i: 0 for i in range(0, dens_level)}
    density_total = {i: 0 for i in range(0, dens_level)}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for idx, pred, label in zip(test_loader.dataset.indices, predicted, labels):
                density_index = min(
                    idx * dens_level // max(test_loader.dataset.indices), dens_level - 1
                )
                density_total[density_index] += 1
                density_correct[density_index] += 1 if pred == label else 0

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate and print accuracy for each fog density level
    density_accuracies = {
        k: (density_correct[k] / density_total[k]) if density_total[k] > 0 else 0
        for k in density_correct.keys()
    }
    for k in sorted(density_accuracies.keys()):
        print(
            f"Fog Density {k}: {density_accuracies[k]:.4f}",
            f"({density_correct[k]}/{density_total[k]})",
        )

    # Plot fog density accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(
        list(density_accuracies.keys()), list(density_accuracies.values()), marker="o"
    )
    plt.title("Accuracy vs. Fog Density")
    plt.xlabel("Fog Density")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Save the plot
    plt.savefig(IMG_PATH + "fog_density_accuracy.png")
    plt.close()


# Updated Plotting function for loss and accuracy
def plot_metrics(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b", label="Training loss")
    plt.plot(epochs, val_losses, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, "g", label="Validation accuracy")
    plt.title("Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(IMG_PATH + "metrics.png")
    plt.close()


# 3. Train the model
train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=EPOCHS)

# 4. Plot the metrics
plot_metrics(train_losses, val_losses, val_accuracies)

# 5. Evaluate the model on the test set with fog density analysis
model.load_state_dict(torch.load(SAVE_WEIGHTS))
evaluate_model_with_fog_density(model, test_loader)
