import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification
from tqdm import tqdm
import wandb

# Initialize Weights & Biases (optional, for logging)
wandb.init(project='vit_finetuning')

# Configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "mps" if torch.backends.mps.is_available() else "cpu"
num_epochs = 10
batch_size = 32
learning_rate = 5e-5

# Paths to your dataset
train_dir = '/Users/neftekhari/Documents/corrected-elastic-egss/train'
val_dir = '/Users/neftekhari/Documents/corrected-elastic-egss/test_copy'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the model
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=3, ignore_mismatched_sizes=True ).to(device)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training loop
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

    # Logging
    wandb.log({
        'Train Loss': train_loss / len(train_loader),
        'Train Accuracy': train_correct / len(train_dataset),
        'Val Loss': val_loss / len(val_loader),
        'Val Accuracy': val_correct / len(val_dataset)
    })

# Save the model
model_path = "vit_finetuned.pt"
torch.save(model.state_dict(), model_path)
model.save_pretrained("/noc/users/noueft/Documents/Code/PIDiff/DiffRes-PI/ViT/ViT-weight/vit_finetuned")


wandb.finish()
