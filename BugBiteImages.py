''' This class is responsible for:
- Converting raw image bytes to a PIL image
- Applying transformations (e.g., resizing, normalization)
- Returning the image and its label in a format suitable for PyTorch models
'''
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform): # CustomDataset class accepts dataframe and transform as parameters.
        self.dataframe = dataframe.reset_index(drop=True)  # Ensures index 0, 1, 2, ...
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx] # Get the row at the given index

        # Extract image and label
        try:
            image_data = row['image']
            label = row['label']
        except KeyError as e:
            print(f"Available columns: {row.index}")
            raise e

        # Decode image bytes into a PIL image
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image = Image.open(BytesIO(image_data['bytes'])).convert("RGB")
        else:
            raise ValueError(f"Unexpected image format at index {idx}: {type(image_data)}")

        # Apply transformations (e.g., resizing, ToTensor, normalization)
        if self.transform:
            image = self.transform(image)

        return image, label