import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification, AutoImageProcessor
from tqdm import tqdm
import os
import wandb

# Import the custom loss functions
from tools.loss import ST_CE_loss, Bal_CE_loss, BCE_loss, LS_CE_loss, MiSLAS_loss, LADE_loss, LDAM_loss, CB_CE_loss

# Initialize Weights & Biases (optional, for logging)
# wandb.init(project='vit_finetuning')

# Configuration
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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

# Calculate the number of samples for training and validation
num_train = int(0.9 * len(full_train_dataset))
num_val = len(full_train_dataset) - num_train

# Split the dataset
train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Hyperparameter grid
learning_rates = [0.0001, 5e-5]
num_epochs_options = [15, 20]

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

def train_and_validate(learning_rate, num_epochs, save_path, loss_type, args):
    # Load the model
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=3,
                                                      ignore_mismatched_sizes=True).to(device)

    # Optimizer and loss
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
            loss = loss_fn(outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                val_correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()

        val_accuracy = val_correct / len(val_dataset)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

            # Save using torch.save
            model_path = os.path.join(save_path, f"vit_finetuned_{loss_type}.pt")
            torch.save(best_model_state, model_path)

            # Save using save_pretrained
            model.load_state_dict(best_model_state)
            model.save_pretrained(os.path.join(save_path, f"vit_finetuned_{loss_type}"))

        # Optional: Add wandb logging here

    return best_val_accuracy

# Ensure the directory exists or create it
save_path = '/noc/users/noueft/Documents/Code/PIDiff/DiffRes-PI/opt-ViT/LAViT/model_weight'
os.makedirs(save_path, exist_ok=True)

# Use the modified function in your grid search
best_hyperparams = None
best_accuracy = 0

loss_type_options = ['ST_CE', 'Bal_CE', 'BCE', 'LS_CE', 'MiSLAS', 'LADE', 'LDAM', 'CB_CE']

# Placeholder for args needed by custom loss functions
class Args:
    cls_num = class_counts.tolist()
    bal_tau = 1.0
    smoothing = 0.1
    nb_classes = 3
    device = device

args = Args()

for loss_type in loss_type_options:
    for lr in learning_rates:
        for epochs in num_epochs_options:
            accuracy = train_and_validate(lr, epochs, save_path, loss_type, args)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparams = (lr, epochs, loss_type)

print(f"Best hyperparameters: Learning Rate = {best_hyperparams[0]}, Num Epochs = {best_hyperparams[1]}, Loss Type = {best_hyperparams[2]}, Accuracy = {best_accuracy}")
