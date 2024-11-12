#!/usr/bin/env python3

import pandas as pd
from PIL import Image
from torchvision.transforms import functional
import argparse
import logging
import torch
import torchvision
import tifffile as tiff
from io import BytesIO
import numpy as np
import os
import csv
logging.basicConfig(level=logging.INFO)
from transformers import AutoImageProcessor, ViTForImageClassification, BeitImageProcessor, BeitForImageClassification

device = "mps" if torch.backends.mps.is_available() else "cpu"
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

def vit_model(model_path, device):
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor

def beit_model(model_path, device):
    processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    model = BeitForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor

def get_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    logging.info(f"Device : {device}")
    return device

def load_model(filename, device, model_version=0, gray=False):
    if model_version == 3:
        return beit_model(filename, device)
    elif model_version == 2:
        return vit_model(filename, device)
    else:
        if gray:
            model = resnet18_gray(num_classes=len(LABELS))
        else:
            if model_version == 0:
                model = resnet50(num_classes=len(LABELS))
            elif model_version == 1:
                model = resnet18(num_classes=len(LABELS))
        model_state_dict = torch.load(filename, map_location="cpu")
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
        return model, None

def classify(image, device, model, processor=None, gray=False):

    if processor:
        image = Image.open(BytesIO(image))
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            _, preds = torch.max(outputs.logits, 1)
    else:
        image = tiff.imread(BytesIO(image))
        t = functional.to_tensor(image)
        t = functional.resize(t, (256, 256))
        t = t.unsqueeze(dim=0)
        if gray:
            t = 0.2125 * t[0, 0, :, :] + 0.7154 * t[0, 1, :, :] + 0.0721 * t[0, 2, :, :]
            t = torch.tile(t, (1, 1, 1, 1))
        t = t.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(t)
            _, preds = torch.max(outputs, 1)
    return LABELS[preds[0]]

def classify_batch(image_list, device, model, processor=None, gray=False, batch_size=1):

    if processor:
        images = [Image.open(BytesIO(image)) for image in image_list]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            _, preds = torch.max(outputs.logits, 1)
        labels = [LABELS[i] for i in preds]
    else:
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
    parser.add_argument("-g", "--gray", action="store_true", help="load lighter weight(Resnet 18) with gray scale")
    parser.add_argument("-m", "--model_version", type=int, default=0, help="model version zero means Resnet50, and 1 means Resnet18, 2 means ViT, 3 means BEiT, default=0")
    parser.add_argument("filename", type=str, help="Path to the tiff file or folder you want to classify")
    parser.add_argument("-f", "--folder", action="store_true", help="folder processing")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size for processing images")
    parser.add_argument("-o", "--output_csv", type=str, help="CSV file to save output in")
    parser.add_argument("-w", "--weights", type=str, help="Optional path to model weights")

    args = parser.parse_args()
    device = get_device()
    weights = args.weights
    if args.model_version == 2:
        if not weights:
            weights = "/Users/neftekhari/Library/CloudStorage/OneDrive-TheAlanTuringInstitute/model-history-on-corrected-dataset/ViT/model_weight/vit_finetuned_MiSLAS_vit_lr5e-05_epochs20"
        model, processor = load_model(weights, device, model_version=args.model_version)

    elif args.model_version == 3:
        if not weights:
            weights = "/Users/neftekhari/Library/CloudStorage/OneDrive-TheAlanTuringInstitute/model-history-on-corrected-dataset/Beit/model_weight/beit_finetuned_MiSLAS_lr5e-05_epochs30"
        model, processor = load_model(weights, device, model_version=args.model_version)
         
    else:
        if args.gray:
            model, processor = load_model("saved_models/resnet18_model_gray_weights_15May.pth", device, gray=True)
        else:
            if args.model_version == 0:
                if not weights:
                    weights = "model.pth"
                model, processor = load_model(weights, device)
            elif args.model_version == 1:
                if not weights:
                    weights =  "saved_models/combined_model.pth"
                model, processor = load_model(weights, device, args.model_version)

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
        results = []
        for i in range(0, len(image_list), args.batch_size):
            batch_images = image_list[i:i + args.batch_size]
            batch_results = classify_batch(batch_images, device, model, processor, args.gray, args.batch_size)
            results.extend(batch_results)

        output_csv_file = args.output_csv
        if not output_csv_file:
            output_csv_file = "/Users/neftekhari/Library/CloudStorage/OneDrive-TheAlanTuringInstitute/model-history-on-corrected-dataset/ViT/output/vit-MiSLAS-vit-5e-05-30.csv"

        with open(output_csv_file, "w", newline="") as csvfile:
            fieldnames = ["Filename", "Predicted Class"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(filenames_list)):
                writer.writerow({"Filename": filenames_list[i], "Predicted Class": results[i]})
        print(f"Results saved to {output_csv_file}")

    else:
        with open(args.filename, "rb") as file:
            image = file.read()
            result = classify(image, device, model, processor, args.gray)
            output_csv_file = "classification_result.csv"
            with open(output_csv_file, "w", newline="") as csvfile:
                fieldnames = ["Filename", "Predicted Class"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"Filename": args.filename, "Predicted Class": result})
            print(f"Result saved to {output_csv_file}")
