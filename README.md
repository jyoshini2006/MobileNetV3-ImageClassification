# MobileNetV3-Large Image Classification

A PyTorch-based image classification project using the **MobileNetV3-Large** architecture with pretrained ImageNet weights.

The project performs image preprocessing, model inference, Softmax probability calculation, and Top-5 image classification.

## Project Overview

This project demonstrates how to perform image classification using **MobileNetV3-Large**.

The model takes an input image, processes it into the required tensor format, runs inference using pretrained weights, and produces the Top-5 predicted ImageNet classes with confidence scores.

The implementation includes the model architecture, pretrained checkpoint loading, image preprocessing, inference, probability calculation, and ImageNet class-name mapping.

## Features

- MobileNetV3-Large architecture implemented in PyTorch
- Pretrained ImageNet weights
- Custom image preprocessing
- RGB image conversion
- Image resizing to `224 × 224`
- ImageNet normalization
- CPU-based inference
- Softmax probability calculation
- Top-5 predictions
- Top-1 prediction with confidence score
- ImageNet class name mapping
- Command-line support for custom image paths

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
│   ├── dog.jpg
│   └── cat.jpg
│
├── weights/
│   └── mobilenetv3-large-1cd25616.pth
│
├── mobilenet.py
├── inference.py
├── imagenet_classes.json
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv312
```

Activate the environment:

```powershell
.venv312\Scripts\activate
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Project

Run inference using the sample dog image:

```powershell
python inference.py inputs/dog.jpg
```

Run inference using the sample cat image:

```powershell
python inference.py inputs/cat.jpg
```

The program displays the Top-5 predicted classes along with their confidence scores and the final Top-1 prediction.

## Running with a Custom Image

You can provide any supported image path through the command line.

```powershell
python inference.py path/to/your/image.jpg
```

For example:

```powershell
python inference.py inputs/dog.jpg
```

The image is loaded and processed before being passed to the MobileNetV3-Large model.

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

## Model Details

- **Architecture:** MobileNetV3-Large
- **Framework:** PyTorch
- **Input Size:** `224 × 224 RGB`
- **Number of Classes:** 1000
- **Pretrained Weights:** ImageNet
- **Inference Device:** CPU
- **Task:** Image Classification

### ImageNet Normalization

The input image is normalized using the standard ImageNet values:

- **Mean:** `[0.485, 0.456, 0.406]`
- **Std:** `[0.229, 0.224, 0.225]`

## Inference Pipeline

The inference process follows these steps:

1. Load the input image.
2. Convert the image to RGB.
3. Resize the image to `224 × 224`.
4. Convert the image to a PyTorch tensor.
5. Scale pixel values from `0–255` to `0–1`.
6. Apply ImageNet normalization.
7. Add the batch dimension.
8. Run MobileNetV3-Large inference.
9. Apply Softmax to obtain class probabilities.
10. Select the Top-5 predictions.
11. Map predicted class indices to ImageNet class names.
12. Display the Top-5 predictions and Top-1 confidence score.

## Pretrained Weights

The project uses a pretrained **MobileNetV3-Large** checkpoint trained on the ImageNet dataset.

The checkpoint is stored locally in:

```text
weights/mobilenetv3-large-1cd25616.pth
```

The pretrained weights are loaded into the MobileNetV3-Large architecture before inference.

## File Description

| File / Folder | Description |
|---|---|
| `mobilenet.py` | Contains the MobileNetV3-Large model architecture |
| `inference.py` | Handles image loading, preprocessing, inference, and prediction |
| `imagenet_classes.json` | Maps ImageNet class indices to class names |
| `weights/` | Contains the pretrained model checkpoint |
| `inputs/` | Contains sample input images |
| `requirements.txt` | Lists required Python dependencies |
| `.gitignore` | Specifies files and folders excluded from Git |
| `LICENSE` | Project license |
| `README.md` | Project documentation |

## Output

For each input image, the program provides:

- Top-5 predicted ImageNet classes
- Probability/confidence for each prediction
- Final Top-1 prediction
- Top-1 confidence score

Example:

```text
1. golden_retriever : 84.95%
2. Labrador_retriever : 3.61%
3. kuvasz : 0.89%
4. Chesapeake_Bay_retriever : 0.69%
5. Brittany_spaniel : 0.38%

Prediction : golden_retriever
Confidence : 84.95%
```

## Requirements

The project requires:

- Python 3.12
- PyTorch
- Pillow
- NumPy

All required packages can be installed using:

```powershell
pip install -r requirements.txt
```

## Notes

- The model performs **classification**, not object detection or segmentation.
- The model predicts one of the **1000 ImageNet classes**.
- The confidence values are obtained from the Softmax probabilities.
- Inference is performed on the **CPU**.
- Input images are converted to RGB before preprocessing.
- The model expects an input resolution of `224 × 224`.

## License

This project is licensed under the **MIT License**.