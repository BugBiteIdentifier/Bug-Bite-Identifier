'''
File to determine the mean and std that is ideal for our dataset
'''
from torch.utils.data import DataLoader
from BugBiteImages import CustomDataset
import torch
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# Load the data from Hugging Face parquet files
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
train_df = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/eceunal/bug-bite-images-aug_v3/" + splits["validation"])

#Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(train_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

mean = torch.zeros(3)
std = torch.zeros(3)
total_images = 0

# Iterate through dataset
for images, _ in tqdm(train_loader, desc="Computing mean/std"):
    mean += images.mean(dim=[0, 2, 3])
    std += images.std(dim=[0, 2, 3])
    total_images += 1

# Compute final mean and std
mean /= total_images
std /= total_images

print("Dataset mean:", mean)
print("Dataset std:", std)