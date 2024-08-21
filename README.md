# ViT-LASNet: Improving in situ real-time classification of long-tail marine plankton images for ecosystem studies

The escalating complexity of image classification tasks in ecological monitoring has highlighted the limitations of conventional models, especially when facing long-tailed data distributions typical of natural environments.

## Overview

This repository presents **ViT-LASNet**, a comprehensive framework designed to improve the classification of plankton images from the Plankton Imaging (Pi-10) dataset, specifically in real-time applications. The model integrates cutting-edge image classification architectures, including Vision Transformers (ViT) and BEiT, along with an innovative dynamic Label-Aware Smoothing (LAS) strategy.

## Key Features

- **Dataset:** Utilizes a novel plankton dataset (Pi-10) for ecological monitoring.
- **Model Architecture:** Implements pre-trained Vision Transformers (ViT) and BEiT models for superior feature extraction.
- **Dynamic Label-Aware Smoothing:** Adjusts smoothing factors based on attention scores to handle long-tailed distributions and improve classification accuracy.
- **Real-Time Application:** Tailored for real-time performance in ecological imaging.

## Motivation

The Pi-10 dataset presents challenges common in ecological datasets, such as imbalanced class distributions. Conventional models struggle under these conditions, motivating the need for approaches like Label-Aware Smoothing, which better handles rare classes.

## Methodology

1. **Data Preprocessing:** Includes standard image augmentation and dataset splitting.
2. **Model Training:** Employs dynamic LAS to adjust model confidence dynamically during training.
3. **Performance Evaluation:** Evaluates the model with and without the Label-Aware Smoothing strategy, demonstrating significant improvements when using this approach.


## Results

The approach showcases marked performance improvements, particularly in handling long-tailed data distributions, setting a new benchmark for ecological image classification.

## Getting Started

### Prerequisites

- Python 3.9.18
- PyTorch
- Required Python libraries (`requirements.txt`)

### Installation

Clone the repository:

```bash
git clone https://github.com/noushineftekhari/ViT-LASNet.git
cd ViT-LASNet
