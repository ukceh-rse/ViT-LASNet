import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification, AutoImageProcessor, BeitForImageClassification, BeitImageProcessor
from tqdm import tqdm
import os

# Import the custom loss functions
from tools.loss import ST_CE_loss, Bal_CE_loss, BCE_loss, LS_CE_loss, MiSLAS_loss, LADE_loss, LDAM_loss, CB_CE_loss

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_workers = 16

# Paths to your dataset
train_dir = '/noc/users/noueft/Documents/corrected-data-elastic-eggs/split-train-test/train'

# Image transformations with data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
class_counts = torch.tensor([full_train_dataset.targets.count(i) for i in range(3)])
class_weights = 1. / class_counts.float()
class_weights = class_weights / class_weights.sum() * 3  # Normalize to sum to number of classes
class_weights = class_weights.to(device)

# Split the dataset
num_train = int(0.9 * len(full_train_dataset))
num_val = len(full_train_dataset) - num_train
train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


# Placeholder for args needed by custom loss functions
class Args:
    cls_num = class_counts.tolist()
    bal_tau = 1.0
    smoothing = 0.1
    nb_classes = 3
    device = device


args = Args()


# Function to one-hot encode labels properly
def one_hot_encode(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels]


# Function to get the loss function based on the selected flag
def get_loss_function(loss_type, class_weights, args):
    if loss_type == 'ST_CE':
        return ST_CE_loss()
    elif loss_type == 'Bal_CE':
        return Bal_CE_loss(args)
    elif loss_type == 'BCE':
        return BCE_loss(args)
    elif loss_type == 'LS_CE':
        return LS_CE_loss(smoothing=args.smoothing)
    elif loss_type == 'MiSLAS':
        return MiSLAS_loss(args)
    elif loss_type == 'LADE':
        return LADE_loss(args)
    elif loss_type == 'LDAM':
        return LDAM_loss(args)
    elif loss_type == 'CB_CE':
        return CB_CE_loss(args)
    else:
        return nn.CrossEntropyLoss()


# Training and validation function
def train_and_validate(model_type, learning_rate, num_epochs, save_path, loss_type, args):
    # Load the model
    if model_type == 'vit':
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", ignore_mismatched_sizes=True,
                                                          num_labels=3).to(device)
    elif model_type == 'beit':
        model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k',
                                                           ignore_mismatched_sizes=True, num_labels=3).to(device)
    else:
        raise ValueError("Unsupported model type. Choose 'vit' or 'beit'.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = get_loss_function(loss_type, class_weights, args)

    best_val_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # One-hot encode labels if needed
            if loss_type in ['ST_CE', 'Bal_CE', 'BCE', 'LADE', 'LDAM', 'CB_CE']:
                labels = one_hot_encode(labels, num_classes=3)

            loss = loss_fn(outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.logits.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                if loss_type in ['ST_CE', 'Bal_CE', 'BCE', 'MiSLAS', 'LADE', 'LDAM', 'CB_CE']:
                    labels = one_hot_encode(labels, num_classes=3)

                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                val_correct += (outputs.logits.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item()

        val_accuracy = val_correct / len(val_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

            # Save the model
            model_path = os.path.join(save_path,
                                      f"{model_type}_finetuned_{loss_type}_lr{learning_rate}_epochs{num_epochs}.pt")
            torch.save(best_model_state, model_path)
            print(f"Saved best model for {model_type} with {loss_type}, lr {learning_rate}, and epochs {num_epochs}")

            # Save using save_pretrained
            model.load_state_dict(best_model_state)
            model.save_pretrained(
                os.path.join(save_path, f"{model_type}_finetuned_{loss_type}_lr{learning_rate}_epochs{num_epochs}"))

    return best_val_accuracy


# Ensure the directory exists or create it
save_path = '/noc/users/noueft/Documents/Code/PIDiff/DiffRes-PI/opt-ViT/train/model_weight'
os.makedirs(save_path, exist_ok=True)

# Grid search over hyperparameters
best_hyperparams = None
best_accuracy = 0
model_type = 'beit'
loss_type_options = ['ST_CE', 'Bal_CE', 'BCE', 'MiSLAS','LDAM', 'CB_CE']
learning_rates = [5e-5]
num_epochs_options = [30]

for loss_type in loss_type_options:
    for lr in learning_rates:
        for epochs in num_epochs_options:
            accuracy = train_and_validate(model_type, lr, epochs, save_path, loss_type, args)
            print(
                f"Model = {model_type}, Learning Rate = {lr}, Num Epochs = {epochs}, Loss Type = {loss_type}, Final Validation Accuracy = {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparams = (model_type, lr, epochs, loss_type)

print(
    f"Best hyperparameters: Model = {best_hyperparams[0]}, Learning Rate = {best_hyperparams[1]}, Num Epochs = {best_hyperparams[2]}, Loss Type = {best_hyperparams[3]}, Accuracy = {best_accuracy}")
