import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# =========================================================
# 1) PATHS
# =========================================================

BASE_DIR = r"C:\Users\anike\OneDrive\Desktop\Frest chirp"

MODEL_1_PATH = os.path.join(BASE_DIR, "model", "bird_model_epoch5.pth")
MODEL_2_PATH = os.path.join(BASE_DIR, "model", "bird_model_epoch8.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
SPECTROGRAM_FOLDER = os.path.join(BASE_DIR, "testing", "spectrogram_testing")

IMG_SIZE = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================================================
# 2) LOAD CLASS NAMES
# =========================================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"class_names.json not found at:\n{CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

print("\nLoaded Classes:")
for i, cls in enumerate(class_names):
    print(f"{i}: {cls}")

# =========================================================
# 3) IMAGE TRANSFORM
# =========================================================
# IMPORTANT:
# Your dataset spectrograms were grayscale 512x512
# So we keep preprocessing same

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Uncomment ONLY if you used normalization during training:
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

# =========================================================
# 4) BUILD MODEL FUNCTION
# =========================================================
# This assumes you trained using ResNet18.
# If you used custom CNN, this must be changed.

def build_model(num_classes):
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# =========================================================
# 5) LOAD BOTH MODELS
# =========================================================

if not os.path.exists(MODEL_1_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_1_PATH}")

if not os.path.exists(MODEL_2_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_2_PATH}")

model_epoch5 = build_model(num_classes)
model_epoch5.load_state_dict(torch.load(MODEL_1_PATH, map_location=device))
model_epoch5 = model_epoch5.to(device)
model_epoch5.eval()

model_epoch8 = build_model(num_classes)
model_epoch8.load_state_dict(torch.load(MODEL_2_PATH, map_location=device))
model_epoch8 = model_epoch8.to(device)
model_epoch8.eval()

print("\nBoth models loaded successfully.")

# =========================================================
# 6) CHECK SPECTROGRAM FOLDER
# =========================================================

if not os.path.exists(SPECTROGRAM_FOLDER):
    raise FileNotFoundError(f"Spectrogram folder not found:\n{SPECTROGRAM_FOLDER}")

image_files = [
    f for f in os.listdir(SPECTROGRAM_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
]

if len(image_files) == 0:
    raise FileNotFoundError(f"No image files found in:\n{SPECTROGRAM_FOLDER}")

print(f"\nFound {len(image_files)} spectrogram images in:")
print(SPECTROGRAM_FOLDER)

# =========================================================
# 7) PREDICTION FUNCTION
# =========================================================

def predict_image(model, image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]

    top3_probs, top3_indices = torch.topk(probs, 3)

    return top3_probs.cpu().numpy(), top3_indices.cpu().numpy()

# =========================================================
# 8) RUN COMPARISON
# =========================================================

print("\n" + "=" * 80)
print("TESTING ALL SPECTROGRAMS ON BOTH MODELS")
print("=" * 80)

for file_name in image_files:
    image_path = os.path.join(SPECTROGRAM_FOLDER, file_name)

    top3_probs_5, top3_indices_5 = predict_image(model_epoch5, image_path)
    top3_probs_8, top3_indices_8 = predict_image(model_epoch8, image_path)

    print("\n" + "=" * 100)
    print(f"IMAGE: {file_name}")
    print("=" * 100)

    # -------------------------
    # MODEL EPOCH 5
    # -------------------------
    print("\nMODEL: bird_model_epoch5.pth")
    print(f"Predicted Bird: {class_names[top3_indices_5[0]]}")
    print(f"Confidence: {top3_probs_5[0]:.4f}")
    print("Top 3 Predictions:")
    for rank in range(3):
        idx = top3_indices_5[rank]
        prob = top3_probs_5[rank]
        print(f"  {rank+1}. {class_names[idx]} -> {prob:.4f}")

    # -------------------------
    # MODEL EPOCH 8
    # -------------------------
    print("\nMODEL: bird_model_epoch8.pth")
    print(f"Predicted Bird: {class_names[top3_indices_8[0]]}")
    print(f"Confidence: {top3_probs_8[0]:.4f}")
    print("Top 3 Predictions:")
    for rank in range(3):
        idx = top3_indices_8[rank]
        prob = top3_probs_8[rank]
        print(f"  {rank+1}. {class_names[idx]} -> {prob:.4f}")

    # -------------------------
    # WINNER
    # -------------------------
    print("\nCOMPARISON:")
    if top3_probs_5[0] > top3_probs_8[0]:
        print("Epoch 5 model is more confident on this image.")
    elif top3_probs_8[0] > top3_probs_5[0]:
        print("Epoch 8 model is more confident on this image.")
    else:
        print("Both models have equal confidence.")