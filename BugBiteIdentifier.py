# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms #This import allows us to access datasets within the torchvision library
# from torch.utils.data import DataLoader
# import pandas as pd
# from BugBiteImages import CustomDataset

# # Define the neural network architecture
# class PracticeCNN(nn.Module):

#     def __init__(self):
#         super(PracticeCNN, self).__init__()
#         # Takes 1 input channel (grayscale), outputs 32 feature maps. Since we are using RGB images, channel is 3.
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

#         self.pool = nn.MaxPool2d(2, 2)

#         #Second convolutional layer takes the 32 feature maps from conv1 and applies 64 new filters
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

#         # Flattened output from conv layers
#         # Input size is 64 * 16 * 64 because that's the shape of the feature map after conv + pooling. Outputs 128 features
#         self.fcLayer1 = nn.Linear(64 * 16 * 64, 128)

#         self.outputLayer = nn.Linear(128, 8)

#     def forward(self, x):
#         # Apply first convolution + ReLU activation + pooling
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)

#         # Apply second convolution + ReLU activation + pooling
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)

#         # Flatten the 3D output into a 1D vector so it can be passed to fully connected layers
#         # `-1` lets PyTorch infer the batch size automatically
#         x = x.view(-1, 64 * 16 * 64)
#         x = torch.relu(self.fcLayer1(x))
#         x = self.outputLayer(x)
#         return x

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Transformation
# transform = transforms.Compose([
#     transforms.Resize((64, 256)),  # Resize to desired input size
#     transforms.ToTensor(),         # Convert to tensor
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the image
# ])

# # Login using e.g. `huggingface-cli login` to access this dataset
# # Our dataset is in parquet format. Parquet stores data in a columnar format
# splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}

# # Load the data from Hugging Face parquet files
# train_df = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["train"])
# test_df = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["validation"])

# train_dataset = CustomDataset(train_df, transform=transform)
# test_dataset = CustomDataset(test_df, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Initialize the model, loss function, and optimizer
# model = PracticeCNN().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # Training function
# def train(model, train_loader, criterion, optimizer, epochs):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(images)

#             # Calculate loss
#             loss = criterion(outputs, labels)
#             loss.backward()

#             # Update weights
#             optimizer.step()

#             # Track loss and accuracy
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         epoch_loss = running_loss / len(train_loader)
#         epoch_accuracy = (correct / total) * 100
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# # Testing function
# def test(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     accuracy = (correct / total) * 100
#     print(f'Test Accuracy: {accuracy:.2f}%')

# # Run the training and testing.
# train(model, train_loader, criterion, optimizer, epochs=15)
# test(model, test_loader)

# torch.save(model.state_dict(), "model/bugbite_cnn_model.pth")
# print("Model saved to model/bugbite_cnn_model.pth")