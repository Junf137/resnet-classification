import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import FAST_DEBUG, IMG_PATH, SAVE_WEIGHTS


# Updated Plotting function for loss and accuracy per epoch
def plot_epoch_loss(epoch, batch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, "b", label="Batch loss")
    plt.title(f"Training loss in Epoch {epoch+1}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if not FAST_DEBUG:
        plt.savefig(IMG_PATH + f"epoch_{epoch+1}_loss.png")

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

    plt.tight_layout()

    if not FAST_DEBUG:
        plt.savefig(IMG_PATH + "metrics.png")

    plt.close()


# Training function
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    train_losses,
    val_losses,
    val_accuracies,
    criterion,
    device,
    num_epochs,
):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_losses = []  # List to store batch losses
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for _, _, inputs, labels in train_loop:
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
        plot_epoch_loss(epoch=epoch, batch_losses=batch_losses)

        # Evaluate the model on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_loop = tqdm(val_loader, total=len(val_loader), leave=True)
            for _, _, inputs, labels in val_loop:
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

        # Save the best model (when not in FAST_DEBUG mode)
        if (val_accuracy > best_val_accuracy) and (not FAST_DEBUG):
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), SAVE_WEIGHTS)


# Evaluation function with fog density analysis
def evaluate_model_with_fog_density(model, test_loader, device, dens_level):
    model.eval()
    correct = 0
    total = 0
    density_correct = {i: 0 for i in range(0, dens_level)}
    density_total = {i: 0 for i in range(0, dens_level)}

    with torch.no_grad():
        # Initialize tqdm
        test_loop = tqdm(test_loader, total=len(test_loader), leave=True)
        for _, fog_lvs, inputs, labels in test_loop:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            max_fog_lv = test_loader.dataset.dataset.max_fog_lv
            for fog_lv, pred, label in zip(fog_lvs, predicted, labels):
                density_index = torch.tensor(
                    min(
                        fog_lv * dens_level // max_fog_lv,
                        dens_level - 1,
                    )
                )
                density_total[density_index.item()] += 1
                density_correct[density_index.item()] += 1 if pred == label else 0

            # Update the progress bar
            test_loop.set_description("Evaluating")
            test_loop.set_postfix(test_accuracy=correct / total)

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

    if not FAST_DEBUG:
        plt.savefig(IMG_PATH + "fog_density_accuracy.png")

    plt.close()
