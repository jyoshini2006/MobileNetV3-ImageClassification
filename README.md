# MobileNetV3-Large Image Classification

A PyTorch-based image classification project using a manually implemented **MobileNetV3-Large** architecture with pretrained ImageNet weights.

The project demonstrates the complete inference pipeline from image preprocessing to Top-5 classification results.

## Project Overview

This project implements MobileNetV3-Large directly in PyTorch and loads a pretrained ImageNet checkpoint into the architecture.

The system accepts an image, preprocesses it according to the ImageNet input requirements, performs inference, converts the output logits into probabilities using Softmax, and displays the Top-5 predicted classes.

The implementation includes the model architecture, pretrained checkpoint loading, image preprocessing, inference, probability calculation, and ImageNet class-name mapping.

## Features

- Manually implemented MobileNetV3-Large architecture
- Pretrained ImageNet checkpoint loading
- Custom image preprocessing
- RGB image conversion
- Image resizing to `224 × 224`
- ImageNet normalization
- CPU-based inference
- Softmax probability calculation
- Top-5 predictions
- Confidence scores
- ImageNet class-name mapping
- Custom image path support

## Model Architecture

MobileNetV3-Large uses several efficient components:

- Inverted Residual blocks
- Depthwise separable convolutions
- Pointwise `1 × 1` convolutions
- Squeeze-and-Excitation (SE) blocks
- Hard-Swish activation
- Hard-Sigmoid activation
- Batch Normalization
- Adaptive Average Pooling

### Model Configuration

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Large |
| Framework | PyTorch |
| Input | `224 × 224 × 3` RGB |
| Output | 1000 classes |
| Parameters | 5,483,032 |
| Dataset | ImageNet |
| Inference | CPU |

## How It Works

The model follows this general process:

```text
Input Image
     ↓
Image Preprocessing
     ↓
MobileNetV3-Large
     ↓
1000 Class Logits
     ↓
Softmax
     ↓
Top-5 Predictions
     ↓
ImageNet Class Names
```

## Workflow

The complete workflow of the project is:

```text
                  ┌──────────────────────┐
                  │      Input Image     │
                  │   dog.jpg / cat.jpg  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Load using PIL    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    RGB Conversion    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Resize 224 × 224   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Tensor Conversion   │
                  │      HWC → CHW       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Normalize Image    │
                  │    ImageNet Mean     │
                  │    ImageNet Std      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  MobileNetV3-Large  │
                  │  Manual Architecture │
                  └──────────┬───────────┘
                             │
                    Pretrained Weights
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Model Output      │
                  │    1000 Logits       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │       Softmax        │
                  │    Probabilities     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Top-5           │
                  │    Predictions       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ ImageNet Class Names │
                  │    + Confidence      │
                  └──────────────────────┘
```

### Workflow Summary

1. **Input:** User provides an image path through the command line.
2. **Loading:** Pillow loads the image and converts it to RGB.
3. **Preprocessing:** The image is resized to `224 × 224`, converted to a tensor, rearranged to CHW format, scaled, and normalized.
4. **Architecture:** The manually implemented MobileNetV3-Large model is created.
5. **Weight Loading:** The pretrained ImageNet checkpoint is loaded into the architecture.
6. **Inference:** The processed image is passed through the model.
7. **Probability:** Softmax converts the output logits into class probabilities.
8. **Classification:** The five highest-probability classes are selected.
9. **Mapping:** Class indices are mapped to ImageNet class names.
10. **Output:** Top-5 predictions and confidence scores are displayed.

## Image Preprocessing

The input image is processed using the following steps:

1. Convert image to RGB.
2. Resize to `224 × 224`.
3. Convert the image to a PyTorch tensor.
4. Convert from HWC to CHW format.
5. Scale pixel values from `0–255` to `0–1`.
6. Apply ImageNet normalization.
7. Add the batch dimension.

### ImageNet Normalization

**Mean:**

```text
[0.485, 0.456, 0.406]
```

**Standard Deviation:**

```text
[0.229, 0.224, 0.225]
```

## Pretrained Weights

The project uses a pretrained **MobileNetV3-Large checkpoint trained on ImageNet**.

The checkpoint is stored locally at:

```text
weights/mobilenetv3-large-1cd25616.pth
```

The pretrained weights are loaded into the manually implemented MobileNetV3-Large architecture using PyTorch's `load_state_dict()`.

## Model Details

- **Architecture:** MobileNetV3-Large
- **Framework:** PyTorch
- **Input Size:** `224 × 224 RGB`
- **Number of Classes:** 1000
- **Inference Device:** CPU
- **Task:** Image Classification

### Model Architecture Summary

- **Total Parameters:** 5,483,032
- **Input Shape:** `1 × 3 × 224 × 224`
- **Output Shape:** `1 × 1000`
- **Final Feature Size:** 960
- **Classifier:** `960 → 1280 → 1000`
- **Global Pooling:** Adaptive Average Pooling
- **Activation:** Hard-Swish / Hard-Sigmoid
- **Attention:** Squeeze-and-Excitation blocks

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

Create a Python virtual environment:

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

## Usage

Run the inference program:

```powershell
python inference.py

## Example Results

### Dog Image

**Input:** `inputs/dog.jpg`

```text
1. golden_retriever : 84.95%
2. Labrador_retriever : 3.61%
3. kuvasz : 0.89%
4. Chesapeake_Bay_retriever : 0.69%
5. Brittany_spaniel : 0.38%
```

**Top-1 Prediction:** `golden_retriever`

**Confidence:** `84.95%`

### Cat Image

**Input:** `inputs/cat.jpg`

```text
1. tiger_cat : 62.85%
2. tabby : 11.54%
3. Egyptian_cat : 7.43%
4. lynx : 0.59%
5. red_fox : 0.48%
```

**Top-1 Prediction:** `tiger_cat`

**Confidence:** `62.85%`

## Validation

The model architecture was validated using a test input with the following dimensions:

```text
Input Shape  : [1, 3, 224, 224]
Output Shape : [1, 1000]
Parameters   : 5,483,032
```

The pretrained checkpoint was successfully loaded and the model successfully performed inference on multiple sample images.

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

## Requirements

The project requires:

- Python 3.12
- PyTorch
- Pillow
- NumPy

All dependencies can be installed using:

```powershell
pip install -r requirements.txt
```

## Notes

- The model performs **image classification**, not object detection or segmentation.
- The model predicts one of the **1000 ImageNet classes**.
- Confidence values are obtained from Softmax probabilities.
- Inference is performed on the CPU.
- Input images are converted to RGB before preprocessing.
- The model expects an input resolution of `224 × 224`.

## Future Improvements

- Add a graphical user interface
- Add GPU/CUDA inference
- Support batch image classification
- Display predictions directly on the uploaded image
- Add inference performance benchmarking
- Add a web-based interface

## License

This project is licensed under the **MIT License**.