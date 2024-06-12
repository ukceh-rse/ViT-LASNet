import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine, RandomResizedCrop, \
    ColorJitter, RandomGrayscale, RandomPerspective, RandomVerticalFlip


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_epochs = 30
data_dir = '/noc/users/noueft/Documents/corrected-data-elastic-eggs/split-train-test'
Save_path='/noc/users/noueft/Documents/corrected-data-elastic-eggs/code/training-versions/045-save-14may-Gray'
# data_dir = '/noc/users/noueft/Documents/elastic-eggs-three-class-split'
# data_dir =  "/Users/neftekhari/Documents/elastic-eggs-dataset/test-dataset"
# Save_path = '/noc/users/noueft/Documents/Code/weighted-loss/New_version_Grey/saved_models'
# Save_path = '/Users/neftekhari/Documents'

def plot_loss(loss_values, current_epoch, num_epochs):
    loss_values = np.array(loss_values)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, current_epoch + 1), loss_values[:current_epoch], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plot_path = os.path.join(Save_path, 'loss_plot-basic.png')
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')


def compute_macro_f1(results, pred_column, true_column, num_classes):
    f1_scores = []
    for i in range(num_classes):
        true_positives = sum((result[pred_column] == i) and (result[true_column] == i) for result in results)
        false_positives = sum((result[pred_column] == i) and (result[true_column] != i) for result in results)
        false_negatives = sum((result[pred_column] != i) and (result[true_column] == i) for result in results)

        precision = true_positives / max((true_positives + false_positives), 1)
        recall = true_positives / max((true_positives + false_negatives), 1)

        f1 = 2 * (precision * recall) / max((precision + recall), 1)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    return macro_f1

def resnet18_gray(num_classes):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet18_rgb(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, num_epochs, num_classes):
    test_results = []
    loss_values = []

    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()

        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

                test_results.extend([{
                    'index': idx * batch_size + i,
                    'true_label': labels[i].item(),
                    'pred': pred[i].item(),
                } for i in range(len(labels))])

                test_loss += loss.item()

        test_loss /= len(test_dataloader.dataset)
        loss_values.append(test_loss)

        plot_loss(loss_values, epoch + 1, num_epochs)

        accuracy = correct / len(test_dataloader.dataset)
        macro_f1 = compute_macro_f1(test_results, 'pred', 'true_label', num_classes)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {test_loss}, Accuracy: {accuracy}, Macro-F1: {macro_f1}')

    #plot_loss(loss_values, num_epochs, num_epochs)

def main(use_gray_model=True):
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomRotation(degrees=30),
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        RandomResizedCrop(size=(224, 224)),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        RandomGrayscale(p=0.1),
        RandomPerspective(distortion_scale=0.2, p=0.5),
        RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
    ])

    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomRotation(degrees=30),
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        RandomResizedCrop(size=(224, 224)),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        RandomGrayscale(p=0.1),
        RandomPerspective(distortion_scale=0.2, p=0.5),
        RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
    ])

    # Use ImageFolder to load training data
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_gray if use_gray_model else transform_rgb)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Class indices:")
    for class_index, class_name in enumerate(train_dataset.classes):
        print(f"{class_name}: {class_index}")


    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform_gray if use_gray_model else transform_rgb)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    # Count the number of samples in each class for class weights
    class_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    # Calculate class weights for the weighted cross-entropy loss
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Choose the model and optimizer based on the color mode
    if use_gray_model:
        model = resnet18_gray(num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        model = resnet18_rgb(num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)


    model.to(device)

    # Train the model
    train_model(model, criterion, optimizer, train_dataloader, test_dataloader, num_epochs, num_classes)

    # Save the trained model's weights
    model_name = 'resnet18_model_gray_weights_14May.pth' if use_gray_model else 'resnet18_model-rgb.pth'
    torch.save(model.state_dict(), model_name)




if __name__ == "__main__":
    # Set the boolean flag to switch between train_dataset_rgb and train_dataset_gray
    use_gray_model = True  # Set to True for grayscale, False for RGB
    main(use_gray_model)


