import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def add_fiducial_markers(image, landmarks, radius=5, color=(0, 255, 0)):
    """
    Add fiducial markers to the image.
    """
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius, color, -1)
    return image

def apply_focus_blur(image, focus_point, blur_radius=10):
    """
    Apply a radial blur to the image, centered around the focus_point.
    """
    # Create a mask for the blur effect
    mask = np.zeros_like(image)
    h, w = mask.shape[:2]
    cv2.circle(mask, focus_point, blur_radius, (255, 255, 255), -1)

    # Apply Gaussian blur to the whole image
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_radius, sigmaY=blur_radius)

    # Combine the blurred and original image
    mask = mask.astype(bool)
    final_image = np.where(mask, blurred_image, image)
    
    return final_image

def augment_image(image_path, landmarks, focus_point):
    image = cv2.imread(image_path)
    
    # Add fiducial markers
    image_with_markers = add_fiducial_markers(image, landmarks)
    
    # Apply focus blur
    augmented_image = apply_focus_blur(image_with_markers, focus_point)
    
    return augmented_image

# Example usage
landmarks = [(50, 50), (100, 100)]  # Example landmarks
focus_point = (75, 75)  # Example focus point

augmented_image = augment_image('img2.jpg', landmarks, focus_point)
cv2.imshow('Augmented Image', augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
