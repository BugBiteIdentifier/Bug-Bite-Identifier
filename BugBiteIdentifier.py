import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms #This import allows us to access datasets within the torchvision library
from torch.utils.data import DataLoader

#The following model will use the MNIST dataset
#The MNIST dataset consists of 70,000 grayscale images of handwritten digits from 0 to 9, each image being a fixed 28x28 pixel square

# Define the neural network architecture
class PracticeCNN(nn.Module):

    def __init__(self):
        super(PracticeCNN, self).__init__()
        # Takes 1 input channel (grayscale), outputs 32 feature maps. If we were using RGB images, the input channel would need to be 3.
        # Applies 32 filters of size 3x3. Padding of 1 keeps the image size at 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Downsamples 28x28 -> 14x14. Not really neccessary for MNIST but helps our NN learn patterns at different scales
        self.pool = nn.MaxPool2d(2, 2)

        #Second convolutional layer takes the 32 feature maps from conv1 and applies 64 new filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # After this and a second pooling layer(done in the forward pass), the image is now 7x7 in size

        # Flattened output from conv layers
        # This is a fully connected layer, also called a dense layer.
        # Input size is 64 * 7 * 7 because that's the shape of the feature map after conv + pooling. Outputs 128 features
        self.fcLayer1 = nn.Linear(64 * 7 * 7, 128)

        # This is our output layer (input: 128, output: 10, for 10 classes of MNIST)
        self.outputLayer = nn.Linear(128, 10)

    def forward(self, x):
        # Apply first convolution + ReLU activation + pooling
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        # Apply second convolution + ReLU activation + pooling
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the 3D output into a 1D vector so it can be passed to fully connected layers
        # `-1` lets PyTorch infer the batch size automatically
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fcLayer1(x))
        x = self.outputLayer(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations. This normalizes the MNIST dataset. This line will need to change if a different dataset is used
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = PracticeCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training function
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (correct / total) * 100
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Testing function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

# Run the training and testing. Just 5 epochs for now, MNIST dataset is very easy, harder datasets will require more.
train(model, train_loader, criterion, optimizer, epochs=5)# Epoch [5/5], Loss: 0.0238, Accuracy: 99.25% <-- This is our training accuracy
test(model, test_loader)#<-- This is the test accuracy