import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import torchvision.transforms as transforms

class FacialLandmarkDataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224), transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.mtcnn = MTCNN(keep_all=True)
        self.transform = transform
        self.image_files, self.landmark_files = self._load_files()
        self.cropped_faces, self.face_landmarks = self._generate_cropped_faces_and_landmarks()
        self.to_tensor = transforms.ToTensor()

    def _load_files(self):
        image_files = {}
        landmark_files = {}
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.jpg'):
                base_filename = filename.split('_')[0]
                img_path = os.path.join(self.root_dir, filename)
                if base_filename not in image_files:
                    image_files[base_filename] = img_path
            if filename.endswith('.pts'):
                base_filename = filename.split('_')[0]
                pts_path = os.path.join(self.root_dir, filename)
                if base_filename not in landmark_files:
                    landmark_files[base_filename] = []
                landmark_files[base_filename].append(pts_path)
        return list(image_files.values()), landmark_files

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image at {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

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
        all_landmarks = np.array(all_landmarks)
        return all_landmarks

    def _crop_faces_and_landmarks(self, img_rgb, landmarks):
        boxes, _ = self.mtcnn.detect(img_rgb)

        cropped_faces = []
        face_landmarks = []

        if boxes is not None and len(landmarks) > 0:
            # We may have more or fewer boxes than landmarks, so we ensure there's a match
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                width, height = x2 - x1, y2 - y1
                margin = 0.4

                # Expand the bounding box slightly to avoid cropping out landmarks
                x1 = max(0, x1 - int(margin * width))
                y1 = max(0, y1 - int(margin * height))
                x2 = min(img_rgb.shape[1], x2 + int(margin * width))
                y2 = min(img_rgb.shape[0], y2 + int(margin * height))

                # Find the landmarks that are within this bounding box
                for landmarks_set in landmarks:
                    # Check if the landmarks correspond to this face by ensuring they are within the bounding box
                    if np.all((landmarks_set[:, 0] >= x1) & (landmarks_set[:, 0] <= x2) & 
                              (landmarks_set[:, 1] >= y1) & (landmarks_set[:, 1] <= y2)):
                        # Adjust landmarks relative to the bounding box
                        adjusted_landmarks = landmarks_set - np.array([x1, y1])

                        # Ensure the landmarks are 68 in number and inside the box
                        if adjusted_landmarks.shape[0] == 68:
                            # Crop the face image
                            face_crop = img_rgb[y1:y2, x1:x2]

                            # Append the face and corresponding landmarks
                            cropped_faces.append(face_crop)
                            face_landmarks.append(adjusted_landmarks)
                        break  # Only consider one set of landmarks per face

        return cropped_faces, face_landmarks


    def _generate_cropped_faces_and_landmarks(self):
        all_cropped_faces = []
        all_face_landmarks = []

        for img_path in self.image_files:
            base_filename = os.path.basename(img_path).split('_')[0]
            pts_paths = self.landmark_files[base_filename]

            img_rgb = self._load_image(img_path)
            landmarks = self._load_landmarks(pts_paths)

            cropped_faces, face_landmarks = self._crop_faces_and_landmarks(img_rgb, landmarks)

            all_cropped_faces.extend(cropped_faces)
            all_face_landmarks.extend(face_landmarks)

        return all_cropped_faces, all_face_landmarks

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
        return len(self.cropped_faces)

    def __getitem__(self, idx):
        img = self.cropped_faces[idx]
        landmarks = self.face_landmarks[idx]

        # Apply transformations if provided
        img_transform, landmarks_transform = self._apply_transform(img, landmarks)

        while len(landmarks_transform) != 68:
            img_transform, landmarks_transform = self._apply_transform(img, landmarks)

        # Convert image to tensor
        img_transform = self.to_tensor(img_transform)  # Convert PIL image to tensor

        # Convert landmarks to tensor
        landmarks_transform = torch.tensor(landmarks_transform, dtype=torch.float32)  # Convert landmarks to tensor

        return img_transform, landmarks_transform
