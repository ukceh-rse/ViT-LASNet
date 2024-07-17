import os
import random
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter
import pandas as pd
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
from urllib3.exceptions import InsecureRequestWarning
from scipy.spatial import distance
import csv


# Suppress the warning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
seed = 42
np.random.seed(seed)


# Define paths

data_dir = "/Users/neftekhari/Desktop/Corrected-dataset/main/nice-nephrop"
save_path = "/Users/neftekhari/Desktop/Corrected-dataset/EDA"

histogram_subfolder = "histogram_plots"
desired_height, desired_width = 128, 128
np.random.seed(42)
# Get class names and counts
class_names = [class_name for class_name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, class_name)) and not class_name.startswith(".")]
num_classes = len(class_names)
class_data = [(class_name, len(os.listdir(os.path.join(data_dir, class_name)))) for class_name in class_names]

# Plot class distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(*zip(*class_data), color='skyblue')
ax.set_title("Class Distribution (Imbalanced)", fontsize=16)
ax.set_xlabel("Class Name", fontsize=12)
ax.set_ylabel("Number of Images", fontsize=12)
ax.tick_params(axis='x', rotation=90)
fig.tight_layout()
ax.set_ylim(0, max(class_data, key=lambda x: int(x[1]))[1] + 100)
# bar_chart_image_path = os.path.join(save_path, "bar_chart.png")
# plt.savefig(bar_chart_image_path)

# Display additional details
counts = list(zip(*class_data))[1]
total_images, min_class_count, max_class_count, mean_class_count, median_class_count = sum(counts), min(counts), max(counts), np.mean(counts), np.median(counts)
print("Total Images:", total_images)
print("Minimum Class Count:", min_class_count)
print("Maximum Class Count:", max_class_count)
print("Mean Class Count:", mean_class_count)
print("Median Class Count:", median_class_count)

# # Create and save class table
table_data = [["Class Name", "Number of Images"]] + class_data
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
table = ax.table(cellText=table_data, loc='center', cellLoc='center', cellColours=[['lightgray'] * 2] + [['white', 'white']] * len(class_data), colLabels=None)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
table_image_path = os.path.join(save_path, "table.png")
plt.savefig(table_image_path, bbox_inches='tight', pad_inches=0.2)

table_data = [["Class Name", "Number of Images"]] + class_data

# Define the path for the CSV file
csv_file_path = os.path.join(save_path, "class_data.csv")

# Save the data to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(table_data)

print(f"CSV file has been saved to {csv_file_path}")

# Function to load and preprocess images in batches
def image_batch_generator(batch_size, class_names):
    images, labels = [], []
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            if not img_name.startswith("."):
                img_path = os.path.join(class_path, img_name)
                image = cv2.resize(cv2.imread(img_path), (desired_width, desired_height))
                mean_color = np.mean(image, axis=(0, 1))
                images.append(image.flatten())
                labels.append(class_name)
                if len(images) == batch_size:
                    yield np.array(images), np.array(labels)
                    images, labels = [], []


# Dimensionality Reduction and Visualization (PCA and t-SNE)
batch_size = 100
pca = PCA(n_components=2)
pca_result = np.empty((0, 2))
# tsne_3D = TSNE(n_components=3, random_state=0)
tsne = TSNE(n_components=2, perplexity=15, n_iter=250)
tsne_result = np.empty((0, 2))
# tsne_3D_result = np.empty((0, 3))


all_labels = []  # Collect all labels

for X_batch, labels_batch in image_batch_generator(batch_size, class_names):
    pca_result_batch = pca.fit_transform(X_batch)
    pca_result = np.concatenate((pca_result, pca_result_batch))

    tsne_result_batch = tsne.fit_transform(X_batch)
    tsne_result = np.concatenate((tsne_result, tsne_result_batch))

    # tsne_3D_batch = tsne_3D.fit_transform(X_batch)
    # tsne_3D_result = np.concatenate((tsne_3D_result, tsne_3D_batch))

    all_labels.extend(labels_batch)  # Collect all labels

# Functions to save PCA and t-SNE visualizations
def save_visualization(result, filename, labels):
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.scatterplot(x=result[:, 0], y=result[:, 1], hue=labels)
    plt.title(f"{filename.capitalize()} Visualization")
    ax.legend(loc='upper right', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename.lower()}_visualization.png"))
    plt.show()

# save_visualization(pca_result, "PCA", labels=all_labels)
save_visualization(tsne_result, "t-SNE", labels=all_labels)




# def save_3d_visualization(result, filename, labels):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Get unique class labels
#     unique_labels = np.unique(labels)
#     num_classes = len(unique_labels)
#
#     # Define colors for each class
#     colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
#
#     # Print unique labels and their colors
#     print("Unique Labels:", unique_labels)
#     print("Colors:", colors)
#
#     # Plot points with different colors for each class
#     for i, label in enumerate(unique_labels):
#         label_indices = np.flatnonzero(labels == label)  # Use np.flatnonzero() instead of np.where()
#         ax.scatter(result[label_indices, 0],
#                    result[label_indices, 1],
#                    result[label_indices, 2],
#                    c=[colors[i]],
#                    label=label)
#
#     ax.set_title(f"{filename.capitalize()} 3D Visualization")
#     ax.legend(loc='upper right')
#
#
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f"{filename.lower()}_3d_visualization.png"))
#     plt.show()
#
# # Example usage:
# save_3d_visualization(tsne_3D_result, "t-SNE", labels=all_labels)


# def find_outliers(tsne_result, threshold=3):
#     # Calculate pairwise distances
#     pairwise_distances = distance.cdist(tsne_result, tsne_result)
#
#     outliers = set()
#
#     for i in range(len(tsne_result)):
#         # Sort distances excluding the point itself
#         sorted_distances = sorted(pairwise_distances[i])[1:]
#
#         # Check if the distance is significantly larger than the neighbors
#         if sorted_distances[-1] > threshold * sorted_distances[-2]:
#             outliers.add(i)
#
#     return list(outliers)
#
# # Assuming tsne_result is your t-SNE result
# outliers = find_outliers(tsne_result)
#
# print("Indices of potential outliers:", outliers)
