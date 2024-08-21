import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(224, 224)):
        self.root_dir = root_dir
        self.image_pairs = self._load_image_pairs()
        self.transform = transform
        self.img_size = img_size
    
    def _load_image_pairs(self):
        image_pairs = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith('_mirror.jpg'):
                normal_img = filename.replace('_mirror.jpg', '.jpg')
                pts_file = filename.replace('_mirror.jpg', '.pts')
                
                mirror_img_path = os.path.join(self.root_dir, filename)
                normal_img_path = os.path.join(self.root_dir, normal_img)
                pts_path = os.path.join(self.root_dir, pts_file)
                
                if os.path.isfile(normal_img_path) and os.path.isfile(pts_path):
                    image_pairs.append((normal_img_path, mirror_img_path, pts_path))
        return image_pairs

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image at {img_path}")
        return img

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
    
    def _resize_image_and_landmarks(self, img, landmarks):
        original_size = img.shape[:2]  # (height, width)
        img_resized = cv2.resize(img, self.img_size)
        
        # Calculate scaling factors
        scale_x = self.img_size[0] / original_size[1]  # new width / old width
        scale_y = self.img_size[1] / original_size[0]  # new height / old height
        
        # Scale the landmarks
        landmarks_resized = landmarks.copy()
        landmarks_resized[:, 0] *= scale_x  # x-coordinates
        landmarks_resized[:, 1] *= scale_y  # y-coordinates
        
        return img_resized, landmarks_resized
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        normal_img_path, mirror_img_path, pts_path = self.image_pairs[idx]
        
        normal_img = self._load_image(normal_img_path)
        mirror_img = self._load_image(mirror_img_path)
        landmarks = self._load_landmarks(pts_path)

        normal_img, landmarks = self._resize_image_and_landmarks(normal_img, landmarks)
        mirror_img, _ = self._resize_image_and_landmarks(mirror_img, landmarks)
        
        if self.transform:
            normal_img = self.transform(normal_img)
            mirror_img = self.transform(mirror_img)
        
        return normal_img, mirror_img, landmarks