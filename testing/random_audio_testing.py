import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# SETTINGS
# -----------------------------
TARGET_FOLDER = "testing/spectogram_testing"
CLASS_NAMES_PATH = "model/class_names.json"

MODEL_EPOCH5 = "model/bird_model_epoch5.pth"
MODEL_EPOCH8 = "model/bird_model_epoch8.pth"

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# -----------------------------
# IMAGE TRANSFORM (must match training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD MODEL FUNCTION
# -----------------------------
def load_model(path):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model_epoch5 = load_model(MODEL_EPOCH5)
model_epoch8 = load_model(MODEL_EPOCH8)

# -----------------------------
# ANALYZE EACH SPECTROGRAM IMAGE
# -----------------------------
print(f"\nComparing Predictions (Epoch 5 vs Epoch 8)")
print(f"{'Chunk':<22} | {'E5 Pred':<15} | {'Conf%':<6} | {'Time(s)':<8} | {'E8 Pred':<15} | {'Conf%':<6} | {'Time(s)':<8}")
print("-" * 120)

files = sorted([
    f for f in os.listdir(TARGET_FOLDER)
    if f.lower().endswith(".png")
])

for file_name in files:
    img_path = os.path.join(TARGET_FOLDER, file_name)

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # -------- Epoch 5 timing --------
        start5 = time.perf_counter()
        out5 = model_epoch5(img)
        torch.cuda.synchronize() if device.type == "cuda" else None
        time5 = time.perf_counter() - start5

        prob5 = F.softmax(out5, dim=1)[0]
        top_prob5, top_idx5 = torch.max(prob5, 0)

        # -------- Epoch 8 timing --------
        start8 = time.perf_counter()
        out8 = model_epoch8(img)
        torch.cuda.synchronize() if device.type == "cuda" else None
        time8 = time.perf_counter() - start8

        prob8 = F.softmax(out8, dim=1)[0]
        top_prob8, top_idx8 = torch.max(prob8, 0)

    pred5 = class_names[top_idx5.item()]
    conf5 = top_prob5.item() * 100

    pred8 = class_names[top_idx8.item()]
    conf8 = top_prob8.item() * 100

    print(f"{file_name:<22} | {pred5:<15} | {conf5:>5.1f}% | {time5:>7.4f} | {pred8:<15} | {conf8:>5.1f}% | {time8:>7.4f}")