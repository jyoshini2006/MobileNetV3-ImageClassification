import sys
import json
import numpy as np
import onnxruntime as ort
from PIL import Image


# ==========================================
# 1. CHECK IMAGE PATH
# ==========================================

if len(sys.argv) < 2:
    print("Usage: python onnx_inference.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]


# ==========================================
# 2. LOAD ONNX MODEL
# ==========================================

model_path = "mobilenetv3_large.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded successfully")


# ==========================================
# 3. LOAD IMAGE
# ==========================================

image = Image.open(image_path).convert("RGB")

print("Image loaded successfully")


# ==========================================
# 4. PREPROCESS IMAGE
# ==========================================

# Resize to MobileNetV3 input size
image = image.resize((224, 224))

# Convert image to NumPy array
image = np.array(image, dtype=np.float32)

# Scale pixels from 0-255 to 0-1
image = image / 255.0

# ImageNet normalization
mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

image = (image - mean) / std

# Convert HWC -> CHW
image = np.transpose(image, (2, 0, 1))

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Make sure the tensor is contiguous
image = np.ascontiguousarray(image, dtype=np.float32)

print("Input shape:", image.shape)


# ==========================================
# 5. RUN ONNX INFERENCE
# ==========================================

input_name = session.get_inputs()[0].name

output = session.run(
    None,
    {input_name: image}
)

predictions = output[0]


# ==========================================
# 6. SOFTMAX
# ==========================================

exp_values = np.exp(
    predictions - np.max(predictions, axis=1, keepdims=True)
)

probabilities = (
    exp_values /
    np.sum(exp_values, axis=1, keepdims=True)
)


# ==========================================
# 7. LOAD IMAGENET CLASS NAMES
# ==========================================

with open("imagenet_classes.json", "r") as f:
    imagenet_classes = json.load(f)


# ==========================================
# 8. TOP-5 PREDICTIONS
# ==========================================

top5_indices = np.argsort(
    probabilities[0]
)[-5:][::-1]

print("\n========== ONNX TOP-5 PREDICTIONS ==========\n")

for i, class_index in enumerate(top5_indices):

    confidence = (
        probabilities[0][class_index] * 100
    )

    class_name = imagenet_classes[str(class_index)]

    print(
        f"{i + 1}. {class_name} : "
        f"{confidence:.2f}%"
    )


# ==========================================
# 9. TOP-1 PREDICTION
# ==========================================

top1_index = top5_indices[0]

prediction = imagenet_classes[str(top1_index)]

confidence = (
    probabilities[0][top1_index] * 100
)

print("\n======================================")
print(f"Prediction : {prediction}")
print(f"Confidence : {confidence:.2f}%")
print("======================================")