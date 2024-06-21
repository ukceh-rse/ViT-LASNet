import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from tqdm import tqdm
import wandb

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

# Calculate the number of samples for training and validation
num_train = int(0.9 * len(full_train_dataset))
num_val = len(full_train_dataset) - num_train

# Split the dataset
train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False)

# Hyperparameter grid
learning_rates = [1e-4, 5e-5, 1e-5]
num_epochs_options = [5, 10, 15]

# Function to train and validate the model
def train_and_validate(learning_rate, num_epochs):
    # Load the model
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=3, ignore_mismatched_sizes=True).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = 0

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

        # Logging
        # wandb.log({
        #     'Learning Rate': learning_rate,
        #     'Epoch': epoch + 1,
        #     'Train Loss': train_loss / len(train_loader),
        #     'Train Accuracy': train_correct / len(train_dataset),
        #     'Val Loss': val_loss / len(val_loader),
        #     'Val Accuracy': val_accuracy
        # })

    return best_val_accuracy

# Perform grid search
best_hyperparams = None
best_accuracy = 0

for lr in learning_rates:
    for epochs in num_epochs_options:
        # wandb.config.update({"learning_rate": lr, "num_epochs": epochs}, allow_val_change=True)
        accuracy = train_and_validate(lr, epochs)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparams = (lr, epochs)

print(f"Best hyperparameters: Learning Rate = {best_hyperparams[0]}, Num Epochs = {best_hyperparams[1]}, Accuracy = {best_accuracy}")

# wandb.finish()
