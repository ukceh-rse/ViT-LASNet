import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from denoising_diffusion_pytorch import Unet

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a Simplified Diffusion Model
class SimpleDiffusionModel(nn.Module):
    def __init__(self, noise_level=0.1):
        super(SimpleDiffusionModel, self).__init__()
        self.noise_level = noise_level
        self.denoising_net = Unet(dim=32, dim_mults=(1, 2, 4)).to(device)  # Smaller network

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_level
        noisy_img = x + noise
        return noisy_img

    def denoise(self, noisy_img, time):
        denoised_img = noisy_img - self.denoising_net(noisy_img, time)
        return denoised_img

# Define the Modified ResNet
class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ModifiedResNet, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, diffusion_model, classification_model):
        super(CombinedModel, self).__init__()
        self.diffusion_model = diffusion_model
        self.classification_model = classification_model

    def forward(self, x):
        time_steps = torch.randint(0, 500, (x.size(0),), device=x.device).long()  # Reduced time steps
        noisy_img = self.diffusion_model(x)
        denoised_img = self.diffusion_model.denoise(noisy_img, time_steps)
        out = self.classification_model(denoised_img)
        return out

# Data transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root='/noc/users/noueft/Documents/corrected-data-elastic-eggs/split-train-test/train', transform=train_transform)
test_dataset = datasets.ImageFolder(
    root='/noc/users/noueft/Documents/corrected-data-elastic-eggs/split-train-test/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)  # Increase batch size, reduce workers
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# Define training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return loss_history

# Define evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Define a function to perform hyperparameter search
def hyperparameter_search(param_grid):
    best_accuracy = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        print(f"Testing with params: {params}")
        # Initialize models
        diffusion_model = SimpleDiffusionModel(noise_level=params['noise_level']).to(device)
        resnet_model = ModifiedResNet(num_classes=3).to(device)
        combined_model = CombinedModel(diffusion_model, resnet_model).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(combined_model.parameters(), lr=params['lr'])

        # Train the model
        train_model(combined_model, train_loader, criterion, optimizer, num_epochs=params['num_epochs'])

        # Evaluate the model
        accuracy = evaluate_model(combined_model, test_loader)
        print(f"Accuracy: {accuracy:.2f}%")

        # Check if we have a new best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"Best Accuracy: {best_accuracy:.2f}% with params: {best_params}")
    return best_params

# Define the hyperparameter grid
param_grid = {
    'num_epochs': [10, 20],
    'lr': [0.001, 0.0001],
    'noise_level': [0.1, 0.2]
}

# Perform hyperparameter search
best_params = hyperparameter_search(param_grid)
print(f"Best Hyperparameters: {best_params}")

# After finding the best hyperparameters, you can train the final model with them
diffusion_model = SimpleDiffusionModel(noise_level=best_params['noise_level']).to(device)
resnet_model = ModifiedResNet(num_classes=3).to(device)
combined_model = CombinedModel(diffusion_model, resnet_model).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=best_params['lr'])

loss_history = train_model(combined_model, train_loader, criterion, optimizer, num_epochs=best_params['num_epochs'])
accuracy = evaluate_model(combined_model, test_loader)

# Plotting function
def plot_loss(loss_history, filename='loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(filename)
    plt.close()

# Save model weights
def save_model_weights(model, path='combined_model.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model weights saved to {path}')

plot_loss(loss_history)
save_model_weights(combined_model)
print(f"Final Model Accuracy: {accuracy:.2f}%")
