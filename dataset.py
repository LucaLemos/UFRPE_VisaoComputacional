import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET

class FacialLandmarkDataset(Dataset):
    def __init__(self, root_dir, xml_path, target_size=(224, 224), transform=None):
        self.root_dir = root_dir
        self.xml_path = xml_path
        self.target_size = target_size
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        # Load images and landmarks from the XML file
        self.image_files, self.face_landmarks = self._load_images_and_landmarks()

    def _load_images_and_landmarks(self):
        image_files = []
        face_landmarks = []

        # Parse the XML file
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        for image in root.findall('images/image'):
            image_file = image.get('file')
            image_path = os.path.join(self.root_dir, image_file)

            landmarks = []
            for part in image.findall('box/part'):
                x = float(part.get('x'))
                y = float(part.get('y'))
                landmarks.append([x, y])

            # Only add images with 68 landmarks
            if len(landmarks) == 68:
                image_files.append(image_path)
                face_landmarks.append(np.array(landmarks, dtype=np.float32))

        return image_files, face_landmarks

    def _apply_transform(self, img, landmarks):
        # Convert image to NumPy array if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    
        # Ensure landmarks are in the format required by albumentations (list of tuples)
        keypoints = [tuple(landmark) for landmark in landmarks]
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=img, keypoints=keypoints)
            img = augmented['image']
            landmarks = np.array(augmented['keypoints'])
        
        # Ensure the image is in NumPy format before resizing
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
    
        # Check if landmarks are still within the image bounds
        h, w = img.shape[:2]

        if len(landmarks) != 68:
            return img, landmarks

        min_x, min_y = np.min(landmarks, axis=0)
        max_x, max_y = np.max(landmarks, axis=0)
    
        # If any landmark is outside the image bounds, expand the image
        if min_x < 0 or min_y < 0 or max_x > w or max_y > h:
            # Compute padding required to fit all landmarks
            pad_x_left = max(0, -min_x)  # Padding needed on the left
            pad_y_top = max(0, -min_y)   # Padding needed on the top
            pad_x_right = max(0, max_x - w)  # Padding needed on the right
            pad_y_bottom = max(0, max_y - h)  # Padding needed on the bottom
    
            # Expand the image using padding, use reflect or replicate padding mode
            img = np.pad(img, 
                         ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)), 
                         mode='reflect')  # 'reflect' or 'edge' (replicate) to avoid constant black padding
    
            # Shift landmarks to adjust for the padding
            landmarks[:, 0] += pad_x_left
            landmarks[:, 1] += pad_y_top
    
        # Resize the image and landmarks if a target size is defined
        if self.target_size:
            original_size = (img.shape[1], img.shape[0])  # (width, height)
            img = cv2.resize(img, self.target_size)
            new_size = (self.target_size[1], self.target_size[0])  # (width, height)
            
            # Scale landmarks accordingly
            scale_x = new_size[0] / original_size[0]
            scale_y = new_size[1] / original_size[1]
            landmarks = landmarks * [scale_x, scale_y]
    
        # Convert image back to PIL for visualization if needed
        img = Image.fromarray(img.astype(np.uint8))  # Ensure image dtype is uint8 before converting to PIL
    
        return img, landmarks

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        landmarks = self.face_landmarks[idx]

        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_transform, landmarks_transform = self._apply_transform(img, landmarks)

        while len(landmarks_transform) != 68:
            img_transform, landmarks_transform = self._apply_transform(img, landmarks)

        # Convert image to tensor
        img_transform = self.to_tensor(img_transform)

        # Convert landmarks to tensor
        landmarks_transform = torch.tensor(landmarks_transform, dtype=torch.float32)

        return img_transform, landmarks_transform
