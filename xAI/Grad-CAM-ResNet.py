# Import necessary libraries
import torch
from torch import nn
from torchvision import models, transforms
from torch.autograd import Function
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2
import os
import random
import matplotlib.gridspec as gridspec

# Define the GradCAM class
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)
        self.model.zero_grad()

        # Calculate gradients
        one_hot = torch.zeros_like(output)
        one_hot[:, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Get the feature maps and gradients
        activations = self.feature_maps
        gradients = self.gradients

        # Calculate CAM
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        return cam

# Define a function to apply Grad-CAM to an input image
def apply_grad_cam(image_path, class_idx):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_array = transform(image)
    input_image = transform(image).unsqueeze(0)

    # Generate CAM
    cam = grad_cam.generate_cam(input_image, class_idx)

    # Visualize the original image and the CAM overlay


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Visualize the original image
    axs[0].imshow(transforms.ToPILImage()(image_array))
    axs[0].set_title('Original Image')
    axs[0].set_aspect('auto')

    # Resize the CAM overlay to match the size of the original image
    resized_cam = transforms.Resize((input_image.shape[2], input_image.shape[3]))(cam)
    axs[1].imshow(transforms.ToPILImage()(input_image.squeeze()))
    pcm = axs[1].imshow(resized_cam.squeeze().detach().numpy(), alpha=0.5, cmap='jet')
    axs[1].set_title('Grad-CAM')
    axs[1].set_aspect('auto')
    cbar = fig.colorbar(pcm, ax=axs[1], shrink=0.6)
    cbar.set_label('Intensity')

    plt.tight_layout()

    fig.savefig(os.path.join(save_dir, os.path.basename(image_path)[:-4] + "_GRAD.png"))

    # plt.show()

# Define the directory containing images for each class
class_directories = {
    0: '/Users/neftekhari/Downloads/Cefas-dataset/test copy/Copepod',
    1: '/Users/neftekhari/Downloads/Cefas-dataset/test copy/Detritus',
    2: '/Users/neftekhari/Downloads/Cefas-dataset/test copy/Non-copepod'
}

# Define a function to randomly select 200 images from each class
def get_random_images(class_directories, num_images=20):
    random_images = []
    for class_idx, directory in class_directories.items():
        images_in_class = os.listdir(directory)
        random_images.extend([(os.path.join(directory, img), class_idx) for img in random.sample(images_in_class, num_images)])
    return random_images



# Specify the path to an image and the corresponding class index
image_path = '/Users/neftekhari/Downloads/Cefas-dataset/test copy'
# image_path = "/Users/neftekhari/Documents/DSG/old-data-set/test"
# save_dir = "/Users/neftekhari/Downloads/Cefas-dataset/Grad-CAM"
# save_dir = "/Users/neftekhari/Documents/DSG/GRAD-CAM"
# save_dir = "/Users/neftekhari/Downloads/Cefas-dataset/DSGEE_Grad-CAM"
save_dir = "/Users/neftekhari/Library/CloudStorage/OneDrive-TheAlanTuringInstitute/trained_model/pre-w-Resnet18/Grad-CAM"
# Load your trained ResNet model
resnet = models.resnet18()
# Assuming your model class has the same architecture as the ResNet model
# If not, replace with your model class and load the weights accordingly
resnet.fc = nn.Linear(resnet.fc.in_features, 3)  # Adjust the last layer for your 3 classes
# resnet.load_state_dict(torch.load('/Users/neftekhari/Library/CloudStorage/OneDrive-TheAlanTuringInstitute/trained_model/Resnet18-RGB-weighted-loss/resnet18_model_rgb_weights.pth', map_location=torch.device('cpu')))
resnet.load_state_dict(torch.load('/Users/neftekhari/PycharmProjects/turing-ecosystem-leadership-award-wp4-plankton/cefas-dsg-repro/src/saved_models/resnet18_model_rgb_weights-pre.pth', map_location=torch.device('cpu')))
# /Users/neftekhari/Downloads/Cefas-dataset/model_18_EE.pth
resnet.eval()
# Specify the target layer for Grad-CAM (last convolutional layer in ResNet)
target_layer = resnet.layer4[-1]
# Initialize GradCam
grad_cam = GradCam(model=resnet, target_layer=target_layer)
grad_cam.register_hooks()


#one sample
# class_idx = 0  # Replace with the index of the class you want to visualize
# Apply Grad-CAM
# apply_grad_cam(image_path, class_idx)


# Get random images from each class
random_images = get_random_images(class_directories)
# Shuffle the list of random images
random.shuffle(random_images)
# Apply Grad-CAM to each random image
for image_path, class_idx in random_images:
    apply_grad_cam(image_path, class_idx)

