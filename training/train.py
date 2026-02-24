import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import json
import os
import time

def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Data Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Best for EfficientNet
    ])

    # 3. Load Data
    train_data = datasets.ImageFolder("Dataset_split/train", transform=transform)
    val_data = datasets.ImageFolder("Dataset_split/val", transform=transform)

    # Windows Fix: num_workers=0 and pin_memory=True for GPU speed
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=16, num_workers=0, pin_memory=True)

    num_classes = len(train_data.classes)
    print("Number of classes:", num_classes)

    # 4. Save Class Names
    os.makedirs("model", exist_ok=True)
    with open("model/class_names.json", "w") as f:
        json.dump(train_data.classes, f)

    # 5. Initialize Model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # 6. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 8

    # 7. Training Loop
    print("\nStarting Training...")
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end='\r')

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{epochs} summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
        print(f"Time: {epoch_time/60:.2f} min\n")

        # 🔥 CHECKPOINT SAVES (ONLY ADDITION)
        if epoch + 1 == 5:
            torch.save(model.state_dict(), "model/bird_model_epoch5.pth")
            print("Checkpoint saved at Epoch 5")

        if epoch + 1 == 8:
            torch.save(model.state_dict(), "model/bird_model_epoch8.pth")
            print("Final checkpoint saved at Epoch 8")

    torch.save(model.state_dict(), "model/bird_model.pth")
    print("Model saved successfully.")

# THIS IS THE CRITICAL WINDOWS GUARD
if __name__ == '__main__':
    train_model()
