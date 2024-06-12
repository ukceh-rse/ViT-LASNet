#!/usr/bin/env python3
import pandas as pd
# Classify images of copepod, non-copepod and detritus.
#
# Run the ResNet50 classifier that was trained on Plankton Analytics
# PIA data supplied by James Scott for the CEFAS Data Study Group at
# the Alan Turing Institute, December 2021.
#
# Based on
# https://github.com/alan-turing-institute/plankton-dsg-challenge/blob/main/notebooks/python/dsg2021/cnn_demo.ipynb
#

from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.transforms import functional
import argparse
import logging
import sys
import torch
import torchvision
import tifffile as tiff
from io import BytesIO
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
device = "mps" if torch.mps else "cpu"
print(f"Using device: {device}")


LABELS = [
    "copepod",
    "detritus",
    "noncopepod",
]


def resnet50(num_classes):
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def resnet18(num_classes):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def resnet18_gray(num_classes):
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device():
    device = torch.device("cpu")

    # Check if GPU is available ..
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"Device : {device}")
    return device


def load_model(filename, device, model_version=0, gray=False):
    if gray:
        model = resnet18_gray(num_classes=len(LABELS))
    else:
        if model_version == 0:
            model = resnet50(num_classes=len(LABELS))
        elif model_version == 1:
            model = resnet18(num_classes=len(LABELS))

    # Load model weights
    model_state_dict = torch.load(filename, map_location="cpu")
    model.load_state_dict(model_state_dict)

    model = model.to(device)

    return model


def classify(image, device, model, gray=False):
    image = tiff.imread(BytesIO(image))

    # Convert Image to tensor and resize it
    t = functional.to_tensor(image)
    t = functional.resize(t, (256, 256))
    t = t.unsqueeze(dim=0)

    if gray:
        t = 0.2125 * t[0, 0, :, :] + 0.7154 * t[0, 1, :, :] + 0.0721 * t[0, 2, :, :]
        t = torch.tile(t, (1, 1, 1, 1))
        # t = t.squeeze(dim=0)

    # Model expects a batch of images so let's convert this image
    # tensor to batch of 1 image

    t = t.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(t)
        # Select the most probable from output
        _, preds = torch.max(outputs, 1)

    return LABELS[preds[0]]


def classify_batch(image_list, device, model, gray, batch_size=1):
    images = [tiff.imread(BytesIO(image)) for image in image_list]

    t_list = [functional.to_tensor(image) for image in images]
    t_resize = [functional.resize(t, (256, 256)) for t in t_list]

    if gray:
        t_resize = [
            0.2125 * t[0, :, :] + 0.7154 * t[1, :, :] + 0.0721 * t[2, :, :]
            for t in t_resize
        ]
        t_resize = [torch.tile(t, (1, 1, 1, 1)) for t in t_resize]
        t_resize = [t.squeeze(dim=0) for t in t_resize]

    t_batch = np.stack(t_resize, axis=0)
    t_batch = torch.from_numpy(t_batch)
    t_batch = t_batch.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(t_batch)
        _, preds = torch.max(outputs, 1)

    labels = [LABELS[i] for i in preds]

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plankton classifier")

    parser.add_argument(
        "-g",
        "--gray",
        action="store_true",
        help="load lighter weight(Resnet 18) with gray scale",
    )

    parser.add_argument(
        "-m",
        "--model_version",
        type=int,
        default=0,
        help="model version zero means Resnet50, and 1 means Resnet18, default=0",
    )

    parser.add_argument(
        "filename", type=str, help="Path to the tiff file you want to classify"
    )

    parser.add_argument(
        "-f",
        "--folder",
        action="store_true",
        help="folder processing",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing images",
    )

    device = get_device()

    args = parser.parse_args()

    if args.gray:
        model = load_model(
            "saved_models/resnet18_model_gray_weights_15May.pth",
            device,
            gray=True,
        )
    else:
        if args.model_version == 0:
            model = load_model("model.pth", device)
        elif args.model_version == 1:
            model = load_model(
                "saved_models/combined_model.pth",
                device,
                args.model_version,
            )

    # Load image or load all images from a folder

    if args.folder:
        image_list = []
        filenames_list = []
        for filename in os.listdir(args.filename):
            if 'tif' in filename:
                filenames_list.append(filename)
                image_path = os.path.join(args.filename, filename)
                with open(image_path, "rb") as file:
                    image = file.read()
                    image_list.append(image)
        # results = (classify_batch(image_list, device, model, args.gray))
        results = []
        for i in range(0, len(image_list), args.batch_size):
            batch_images = image_list[i:i + args.batch_size]
            batch_results = classify_batch(batch_images, device, model, args.gray, args.batch_size)
            results.extend(batch_results)


    #     for i in range(filenames_list.__len__()):
    #         print(f"{filenames_list[i]:<20}{results[i]:<20}")
    #
    # else:
    #     with open(args.filename, "rb") as file:
    #         image = file.read()
    #         print(classify(image, device, model, args.gray))

            # Save results to a CSV file



        output_csv_file = "/Users/neftekhari/Library/CloudStorage/OneDrive-TheAlanTuringInstitute/trained_model/DiffRes/classification results/combined_model_21_May.csv"
        with open(output_csv_file, "w", newline="") as csvfile:
            fieldnames = ["Filename", "Predicted Class"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(filenames_list)):
                writer.writerow({"Filename": filenames_list[i], "Predicted Class": results[i]})

        print(f"Results saved to {output_csv_file}")

    # else:
    #     with open(args.filename, "rb") as file:
    #         image = file.read()
    #         result = classify(image, device, model, args.gray)
    #
    #         # Save result to a CSV file
    #         output_csv_file = "classification_result.csv"
    #         with open(output_csv_file, "w", newline="") as csvfile:
    #             fieldnames = ["Filename", "Predicted Class"]
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #             writer.writeheader()
    #             writer.writerow({"Filename": args.filename, "Predicted Class": result})
    #
    #         print(f"Result saved to {output_csv_file}")



# -m 1 -f -b 200  /Users/neftekhari/Documents/Dataset