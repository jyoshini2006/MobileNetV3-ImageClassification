import torch
from PIL import Image
import json
from mobilenet import mobilenetv3_large


# ==========================================
# 1. CREATE MOBILENETV3-LARGE
# ==========================================

model = mobilenetv3_large(num_classes=1000)

print("MobileNetV3-Large architecture created")


# ==========================================
# 2. LOAD PRETRAINED WEIGHTS
# ==========================================

weights_path = "weights/mobilenetv3-large-1cd25616.pth"

checkpoint = torch.load(weights_path, map_location="cpu")

# Handle either direct state_dict or checkpoint containing state_dict
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)

print("Pretrained weights loaded successfully")


# ==========================================
# 3. EVALUATION MODE
# ==========================================

model.eval()

print("Model is ready for inference")


# ==========================================
# 4. LOAD IMAGE
# ==========================================

import sys

if len(sys.argv) < 2:
    print("Usage: python inference.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")

print("Image loaded successfully")


# ==========================================
# 5. PREPROCESS IMAGE
# ==========================================

image = image.resize((224, 224))

# Convert PIL image -> PyTorch tensor
image = torch.tensor(
    list(image.getdata()),
    dtype=torch.float32
)

image = image.reshape(224, 224, 3)

# Convert HWC -> CHW
image = image.permute(2, 0, 1)

# Scale pixels from 0-255 to 0-1
image = image / 255.0


# ImageNet normalization
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)

image = (image - mean) / std


# Add batch dimension
image = image.unsqueeze(0)


# ==========================================
# 6. RUN INFERENCE
# ==========================================

with torch.no_grad():
    output = model(image)


# ==========================================
# 7. SOFTMAX
# ==========================================

probabilities = torch.softmax(output, dim=1)


# ==========================================
# 8. LOAD IMAGENET CLASS NAMES
# ==========================================

with open("imagenet_classes.json", "r") as f:
    imagenet_classes = json.load(f)


# ==========================================
# 9. TOP-5 PREDICTIONS
# ==========================================

top5_prob, top5_indices = torch.topk(
    probabilities,
    5
)

print("\n========== TOP-5 PREDICTIONS ==========\n")

for i in range(5):

    class_index = top5_indices[0][i].item()

    confidence = top5_prob[0][i].item() * 100

    class_name = imagenet_classes[str(class_index)]

    print(
        f"{i + 1}. {class_name} : "
        f"{confidence:.2f}%"
    )


# ==========================================
# 10. TOP-1 PREDICTION
# ==========================================

top1_index = top5_indices[0][0].item()

prediction = imagenet_classes[str(top1_index)]

confidence = top5_prob[0][0].item() * 100


print("\n======================================")
print(f"Prediction : {prediction}")
print(f"Confidence : {confidence:.2f}%")
print("======================================")