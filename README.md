# MobileNetV3-Large Image Classification

A PyTorch-based image classification project using the **MobileNetV3-Large** architecture with pretrained ImageNet weights.

The project performs image preprocessing, model inference, Softmax probability calculation, and Top-5 image classification.

## Project Overview

This project demonstrates how to perform image classification using MobileNetV3-Large.

The model takes an input image, processes it into the required tensor format, runs inference using pretrained weights, and produces the Top-5 predicted ImageNet classes with confidence scores.

## Features

- MobileNetV3-Large architecture implemented in PyTorch
- Pretrained ImageNet weights
- Custom image preprocessing
- ImageNet normalization
- CPU-based inference
- Softmax probability calculation
- Top-5 predictions
- Top-1 prediction with confidence score
- ImageNet class name mapping

## Technologies Used

- Python 3.12
- PyTorch
- Pillow (PIL)
- NumPy
- MobileNetV3-Large
- ImageNet

## Project Structure

```text
MobileNetV3-ImageClassification/
│
├── inputs/
│   └── dog.jpg
│
├── weights/
│   └── mobilenetv3-large-1cd25616.pth
│
├── mobilenet.py
├── inference.py
├── imagenet_classes.json
├── .gitignore
└── README.md
## Running with a Custom Image

You can provide any image path through the command line.

```powershell
python inference.py inputs/dog.jpg
## Example Results

### Dog Image

- **Input:** `inputs/dog.jpg`
- **Prediction:** `golden_retriever`
- **Confidence:** `84.95%`

### Cat Image

- **Input:** `inputs/cat.jpg`
- **Prediction:** `tiger_cat`
- **Confidence:** `62.85%`

### Top-5 Predictions

For `dog.jpg`:

| Rank | Class | Confidence |
|------|-------|------------|
| 1 | golden_retriever | 84.95% |
| 2 | Labrador_retriever | 3.61% |
| 3 | kuvasz | 0.89% |
| 4 | Chesapeake_Bay_retriever | 0.69% |
| 5 | Brittany_spaniel | 0.38% |