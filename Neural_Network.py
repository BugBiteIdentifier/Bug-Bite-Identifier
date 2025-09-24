import pandas as pd
from PIL import Image
import io
import numpy as np


# load dataset from huggingface
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["train"])
df_val = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["validation"])

# load images as bytes, and labels as ints
X_t, y_train = df['image'], df['label']
X_val, y_test = df_val['image'], df_val['label']

# convert images from bytes to images
def bytes_dict_to_pil_image(image_dict):
    image_bytes = image_dict['bytes']
    image_stream = io.BytesIO(image_bytes)
    return Image.open(image_stream)

# convert training and validation images
X_train = X_t.apply(bytes_dict_to_pil_image)
X_test = X_val.apply(bytes_dict_to_pil_image)

# convert images to 1D arrays based on pixels
def convert_image(img):
    img = img.convert("RGB")     # make sure all images are the same format   
    img = img.resize((224, 224)) # and size
    tmp = np.array(img)
    return (tmp.flatten()) / 255.0  # divide by 255 to normalize pixels between 0 and 1

# convert training and validation images
X_train = np.array(X_train.apply(convert_image))
X_test = np.array(X_test.apply(convert_image))


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

import torch


X_train = torch.tensor(np.stack(X_train), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(np.stack(X_test), dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

input_size = X_train.shape[1]
hidden_size = 100
num_classes = 8

model = MLP(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import TensorDataset, DataLoader

batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 4

for epoch in range(num_epochs):
  for batch_X, batch_y in train_loader:
    #Forward Pass
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    #Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
  outputs = model(X_test)
  _, predicted = torch.max(outputs, 1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predicted)
print(f'Accuracy: {accuracy * 100:.2f}')
