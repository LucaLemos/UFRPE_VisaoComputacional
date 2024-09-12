import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FacialLandmarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self._load_image_pairs()  # Update to load image pairs

    def _load_image_pairs(self):
        image_pairs = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.jpg') and '_mirror' not in filename:
                normal_img_path = os.path.join(self.root_dir, filename)
                pts_file = filename.replace('.jpg', '.pts')
                pts_path = os.path.join(self.root_dir, pts_file)
                
                if os.path.isfile(pts_path):
                    image_pairs.append((normal_img_path, pts_path))
        
        return image_pairs

    def _load_landmarks(self, pts_path):
        with open(pts_path, 'r') as file:
            lines = file.readlines()
        points = []
        reading_points = False
        for line in lines:
            line = line.strip()
            if line.startswith('{'):
                reading_points = True
                continue
            if reading_points:
                if line.startswith('}'):
                    break
                x, y = map(float, line.split())
                points.append((x, y))
        return np.array(points)

    def _transform_landmarks(self, landmarks, original_width, original_height):
        # Apply scaling to landmarks
        scale_x = 224 / original_width
        scale_y = 224 / original_height
        landmarks = [(x * scale_x, y * scale_y) for x, y in landmarks]
        return np.array(landmarks)

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img_path, pts_path = self.image_pairs[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        original_width, original_height = img.size
        
        # Load landmarks
        landmarks = self._load_landmarks(pts_path)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        # Adjust landmarks to the transformed image size
        if self.transform:
            img_transformed_size = (224, 224)  # Assuming resize transformation to 224x224
            landmarks = self._transform_landmarks(landmarks, original_width, original_height)

        landmarks = torch.tensor(landmarks, dtype=torch.float32).view(-1)
        
        return img, landmarks