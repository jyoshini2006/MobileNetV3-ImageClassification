import torch
from mobilenet import mobilenetv3_large

# 1. Create MobileNetV3-Large
model = mobilenetv3_large(num_classes=1000)

print("MobileNetV3-Large architecture created")

# 2. Load pretrained weights
weights_path = "weights/mobilenetv3-large-1cd25616.pth"

checkpoint = torch.load(
    weights_path,
    map_location="cpu"
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)

print("Pretrained weights loaded successfully")

# 3. Evaluation mode
model.eval()

# 4. Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

print("Dummy input created:", dummy_input.shape)

# 5. Export PyTorch model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv3_large.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"]
)

print("ONNX model exported successfully!")
print("Saved as: mobilenetv3_large.onnx")