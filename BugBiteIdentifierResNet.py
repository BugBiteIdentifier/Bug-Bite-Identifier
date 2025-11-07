import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
from BugBiteImages import CustomDataset
import os
# Enable MPS fallback for compatibility for operations that don't support MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' 

print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())


# Check for MPS (Apple Silicon), then CUDA, then fall back to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to input size
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=(0.6701, 0.5235, 0.4636), std=(0.2392, 0.2142, 0.2095))  # Normalize the image
])

# Login using e.g. `huggingface-cli login` to access this dataset
# Our dataset is in parquet format. Parquet stores data in a columnar format
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}

# Load the data from Hugging Face parquet files
train_df = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["validation"])

train_dataset = CustomDataset(train_df, transform=transform)
test_dataset = CustomDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class ResNet18(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

resnet_model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for i, (images, labels) in enumerate(loader, 1):  # start counting from 1
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Print every 10 batches
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(loader)}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    epoch_precision = precision_score(all_labels, all_preds, average='macro')
    epoch_recall = recall_score(all_labels, all_preds, average='macro')
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    epoch_precision = precision_score(all_labels, all_preds, average='macro')
    epoch_recall = recall_score(all_labels, all_preds, average='macro')
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1


num_epochs = 50
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc, train_precision, train_recall, train_f1 = train(resnet_model, train_loader, criterion, optimizer)
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(resnet_model, test_loader, criterion)
    print(f"[Epoch {epoch}/{num_epochs}] "
          f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    
# Save the model
torch.save(resnet_model.state_dict(), "model/bugbite_cnn_model.pth")
print("Model saved to model/bugbite_cnn_model.pth")