import deeplake
import numpy as np
import os

# Load dataset from DeepLake
try:
    ds = deeplake.load("hub://activeloop/300w")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Create directories for saving images and keypoints
os.makedirs('images', exist_ok=True)
os.makedirs('keypoints', exist_ok=True)

for i in range(len(ds['images'])):
    image = ds['images'][i].numpy()
    landmarks = ds['keypoints'][i].numpy()
    
    # Ensure image is uint8 type
    image = image.astype('uint8')

    # Convert grayscale to RGB if needed
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    
    if landmarks.shape[1] == 1:
        landmarks = landmarks.reshape(-1, 2)
    
    # Save image and keypoints
    np.savez_compressed(f'images/image_{i}.npz', image=image)
    np.savez_compressed(f'keypoints/keypoints_{i}.npz', keypoints=landmarks)

print("Images and keypoints saved locally.")