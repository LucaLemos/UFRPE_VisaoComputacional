import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SiameseDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_pairs = self._load_image_pairs()
    
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
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        normal_img_path, mirror_img_path, pts_path = self.image_pairs[idx]
        
        normal_img = self._load_image(normal_img_path)
        mirror_img = self._load_image(mirror_img_path)
        landmarks = self._load_landmarks(pts_path)
        
        return normal_img, mirror_img, landmarks
