import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
import json
import time  # <-- added

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load validation dataset
val_data = datasets.ImageFolder("Dataset_split/val", transform=transform)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

num_classes = len(val_data.classes)

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

def evaluate_model(model_path):
    print(f"\nEvaluating: {model_path}")

    start_time = time.time()  # <-- added

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    elapsed_time = time.time() - start_time  # <-- added

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Time Taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")  # <-- added

# Evaluate both checkpoints
evaluate_model("model/bird_model_epoch5.pth")
evaluate_model("model/bird_model_epoch8.pth")
