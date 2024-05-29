import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ResNetForImageClassification
import torchvision.transforms as transforms

from partition import get_dataloader
from training import train_model, plot_metrics, evaluate_model_with_fog_density

# global variables
DENS_LEVEL = 10
PARTITION_VERSION = 2  # 1 or 2

BASE_FOLDER = "../_DatasetsLocal/CompoundEyeClassification/Data/"
MODEL_PATH = "../_ModelsLocal/resnet-50"
SAVE_WEIGHTS = "./models/best_model_ver_" + str(PARTITION_VERSION) + ".pth"
IMG_PATH = "./images/ver_" + str(PARTITION_VERSION) + "/"
CONTINUE_TRAINING = True

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_CLASSES = 3
TRAINING_PORTION = 0.8
VALIDATION_PORTION = 0.1
EPOCHS = 2


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


# 1. Create datasets & Data preprocessing
train_loader, val_loader, test_loader = get_dataloader(
    base_folder=BASE_FOLDER,
    ver=PARTITION_VERSION,
    train_portion=TRAINING_PORTION,
    val_portion=VALIDATION_PORTION,
    batch_size=BATCH_SIZE,
    transform=transform,
)


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


# 3. Train the model
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    train_losses=train_losses,
    val_losses=val_losses,
    val_accuracies=val_accuracies,
    criterion=criterion,
    device=device,
    num_epochs=EPOCHS,
    save_weights=SAVE_WEIGHTS,
    save_img_path=IMG_PATH,
)

# 4. Plot the metrics
plot_metrics(
    train_losses=train_losses,
    val_losses=val_losses,
    val_accuracies=val_accuracies,
    save_img_path=IMG_PATH,
)

# 5. Evaluate the model on the test set with fog density analysis
model.load_state_dict(torch.load(SAVE_WEIGHTS))
evaluate_model_with_fog_density(
    model=model,
    test_loader=test_loader,
    device=device,
    dens_level=DENS_LEVEL,
    save_img_path=IMG_PATH,
)
