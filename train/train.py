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
import wandb

# Import the custom loss functions
from tools.loss import ST_CE_loss, Bal_CE_loss, BCE_loss, LS_CE_loss, MiSLAS_loss, LADE_loss, LDAM_loss, CB_CE_loss, \
    MiSLAS_vit_loss, BCE_CL_loss

# Initialize Weights & Biases (optional, for logging)
# wandb.init(project='vit_finetuning')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Configuration
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_workers = 16

# Paths to your dataset
train_dir = '/noc/users/noueft/Documents/New-taxonomy-model/train'
save_path = '/noc/users/noueft/Documents/New-taxonomy-model/Code/train/model_weight'
os.makedirs(save_path, exist_ok=True)

# Image transformations with data augmentation
transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.Resize(256),  # Resize smaller side to 256, maintaining aspect ratio
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)

# Print the class names and their corresponding indices
print("Class indices and labels:")
for idx, class_name in enumerate(full_train_dataset.classes):
    print(f"{idx}: {class_name}")

class_counts = torch.tensor([full_train_dataset.targets.count(i) for i in range(18)])
class_weights = 1. / class_counts.float()
class_weights = class_weights / class_weights.sum() * 18  # Normalize to sum to number of classes
class_weights = class_weights.to(device)

# Calculate the number of samples for training and validation
num_train = int(0.9 * len(full_train_dataset))
num_val = len(full_train_dataset) - num_train

# Split the dataset
train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Hyperparameter grid
learning_rates = [5e-5]
num_epochs_options = [20, 30]


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
    elif loss_type == 'MiSLAS_vit':
        return MiSLAS_vit_loss(args)
    elif loss_type == 'BCE_CL':
        return BCE_CL_loss(args)
    elif loss_type == 'LADE':
        return LADE_loss(args)
    elif loss_type == 'LDAM':
        return LDAM_loss(args)
    elif loss_type == 'CB_CE':
        return CB_CE_loss(args)
    else:
        return nn.CrossEntropyLoss()


def one_hot_encode(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels]


def train_and_validate(model_type, learning_rate, num_epochs, save_path, loss_type, args):
    # Load the model
    if model_type == 'vit':
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=18,
                                                          output_attentions=True, ignore_mismatched_sizes=True).to(
            device)
    elif model_type == 'beit':
        processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
        model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k',
                                                           output_attentions=True, ignore_mismatched_sizes=True,
                                                           num_labels=18).to(device)
    else:
        raise ValueError("Unsupported model type. Choose 'vit' or 'beit'.")

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

            # One-hot encode labels if needed
            if loss_type in ['ST_CE', 'Bal_CE', 'BCE', 'MiSLAS', 'MiSLAS_vit', 'LADE', 'LDAM', 'CB_CE', 'BCE_CL']:
                labels = one_hot_encode(labels, num_classes=18)

            # Calculate loss with attention scores if applicable
            if loss_type == 'MiSLAS_vit':
                loss = loss_fn(outputs.logits, labels, outputs.attentions)
            else:
                loss = loss_fn(outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.logits.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item() if loss_type in [
                'ST_CE', 'Bal_CE', 'BCE', 'MiSLAS', 'MiSLAS_vit', 'LADE', 'LDAM', 'CB_CE', 'BCE_CL'] else (
                    outputs.logits.argmax(dim=-1) == labels).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # One-hot encode labels if needed
                if loss_type in ['ST_CE', 'Bal_CE', 'BCE', 'MiSLAS', 'MiSLAS_vit', 'LADE', 'LDAM', 'CB_CE', 'BCE_CL']:
                    labels = one_hot_encode(labels, num_classes=18)

                # Calculate loss with attention scores if applicable
                if loss_type == 'MiSLAS_vit':
                    loss = loss_fn(outputs.logits, labels, outputs.attentions)
                else:
                    loss = loss_fn(outputs.logits, labels)

                val_loss += loss.item()
                val_correct += (outputs.logits.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item() if loss_type in [
                    'ST_CE', 'Bal_CE', 'BCE', 'MiSLAS', 'MiSLAS_vit', 'LADE', 'LDAM', 'CB_CE', 'BCE_CL'] else (
                        outputs.logits.argmax(dim=-1) == labels).sum().item()

        val_accuracy = val_correct / len(val_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

            # Save using torch.save
            model_path = os.path.join(save_path,
                                      f"{model_type}_finetuned_{loss_type}_lr{learning_rate}_epochs{num_epochs}.pt")
            torch.save(best_model_state, model_path)
            print(f"Saved best model for {model_type} with {loss_type}, lr {learning_rate}, and epochs {num_epochs}")

            # Save using save_pretrained
            model.load_state_dict(best_model_state)
            model.save_pretrained(
                os.path.join(save_path, f"{model_type}_finetuned_{loss_type}_lr{learning_rate}_epochs{num_epochs}"))

    return best_val_accuracy


best_hyperparams = None
best_accuracy = 0

loss_type_options = ['BCE_CL']


class Args:
    cls_num = class_counts.tolist()
    bal_tau = 1.0
    smoothing = 0.1
    nb_classes = 18
    device = device


args = Args()

model_type = 'vit'  # Change to 'beit' to use the BEiT model

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
