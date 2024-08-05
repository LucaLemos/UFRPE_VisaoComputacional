import deeplake
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Load the dataset from DeepLake
def load_deeplake_dataset():
    ds = deeplake.load("hub://activeloop/300w")
    return ds

class DeeplakeDataset(Dataset):
    def __init__(self, deeplake_ds, transform=None):
        self.deeplake_ds = deeplake_ds
        self.transform = transform

    def __len__(self):
        return len(self.deeplake_ds['images'])

    def __getitem__(self, idx):
        # Fetch image, keypoints, and labels from DeepLake dataset
        image = self.deeplake_ds['images'][idx].numpy()  # Convert to numpy array
        keypoints = self.deeplake_ds['keypoints'][idx].numpy()
        label = self.deeplake_ds['labels'][idx].numpy()

        if keypoints.shape[1] == 1:
            keypoints = keypoints.reshape(-1, 2)

        # Convert numpy array to PIL Image for transformation
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, keypoints, label

def get_dataloader(batch_size=32):
    # Load dataset
    ds = load_deeplake_dataset()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    # Create dataset and DataLoader
    dataset = DeeplakeDataset(ds, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Adjust num_workers as needed

    return dataloader