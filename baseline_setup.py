import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# --- Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS_FINETUNE = 3
BATCH_SIZE = 64
LR = 0.0001
MODEL_PATH_FP32_FULL = "models/mobilenet_v3_full_fp32.pth"
MODEL_PATH_FP32_EDGE = "models/edge_model_fp32.pth"
MODEL_PATH_FP32_SERVER = "models/server_model_fp32.pth"

# --- 1. Load Data ---
def get_dataloaders():
    print("Loading CIFAR-100 data...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return trainloader, testloader

# --- 2. Load and Fine-Tune Model ---
def get_model():
    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 100)
    )
    return model.to(DEVICE)

def test_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- 3. Split the Model ---
def split_and_verify_model(full_model):
    edge_model = full_model.features.to(DEVICE).eval()
    server_model = nn.Sequential(
        full_model.avgpool,
        nn.Flatten(1),
        full_model.classifier
    ).to(DEVICE).eval()
    return edge_model, server_model
