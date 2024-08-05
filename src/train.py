import torch
import torch.optim as optim
from model import SiameseNetwork
from loss import DCCALoss
from dataset import get_dataloader
from augmentation import fiducial_focus_augmentation

def train(num_epochs=10, batch_size=32, lr=0.001):
    dataloader = get_dataloader(batch_size=batch_size)

    model = SiameseNetwork()
    criterion = DCCALoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for images, landmarks, labels in dataloader:
            # Apply augmentation
            augmented_images = [fiducial_focus_augmentation(img.numpy(), lm.numpy()) for img, lm in zip(images, landmarks)]
            augmented_images = torch.stack([torch.tensor(img).float() for img in augmented_images])

            # Forward pass
            output1, output2 = model(augmented_images, augmented_images)
            loss = criterion(output1, output2)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

if __name__ == "__main__":
    train()