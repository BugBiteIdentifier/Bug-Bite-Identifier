import torch
import torch.nn as nn
from torchvision import models

class PracticeCNN(nn.Module):
    def __init__(self):
        super(PracticeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.fcLayer1 = nn.Linear(64*16*64,128)
        self.outputLayer = nn.Linear(128,8)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*16*64)
        x = torch.relu(self.fcLayer1(x))
        x = self.outputLayer(x)
        return x
    
#added this model based on the ResNet 18 code from BugBiteIdentifierResNet.py
class ResNet18(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
