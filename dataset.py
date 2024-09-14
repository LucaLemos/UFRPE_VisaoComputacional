import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class FacialLandmarkDataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.image_files, self.landmark_files = self._load_files()
    
    def _load_files(self):
        image_files = {}
        landmark_files = {}

        # Loop through the directory and collect image files and their corresponding landmarks
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.jpg'):
                base_filename = filename.split('_')[0]  # Keep only the base of the filename (without _1, _2)
                img_path = os.path.join(self.root_dir, filename)
                if base_filename not in image_files:
                    image_files[base_filename] = img_path
                
            if filename.endswith('.pts'):
                base_filename = filename.split('_')[0]  # Keep the same base to associate with the image
                pts_path = os.path.join(self.root_dir, filename)
                if base_filename not in landmark_files:
                    landmark_files[base_filename] = []
                landmark_files[base_filename].append(pts_path)

        return list(image_files.values()), landmark_files

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image at {img_path}")
        # Store original size
        self.image_size = (img.shape[1], img.shape[0])
        # Resize the image
        img = cv2.resize(img, self.target_size)
        return img

    def _load_landmarks(self, pts_paths):
        all_landmarks = []
        for pts_path in pts_paths:
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
            all_landmarks.append(np.array(points))
        
        # Convert landmarks to a single array
        all_landmarks = np.array(all_landmarks)
        
        # Resize landmarks if they exist
        if all_landmarks.size > 0:
            # Calculate scaling factors
            x_scale = self.target_size[0] / self.image_size[0]
            y_scale = self.target_size[1] / self.image_size[1]
            
            # Scale landmarks
            scaled_landmarks = []
            for landmarks in all_landmarks:
                scaled_landmarks.append(landmarks * [x_scale, y_scale])
            return np.array(scaled_landmarks)
        return np.array([])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_filename = os.path.basename(img_path).split('_')[0]
        pts_paths = self.landmark_files[base_filename]
        
        img = self._load_image(img_path)
        landmarks = self._load_landmarks(pts_paths)
        
        return img, landmarks
