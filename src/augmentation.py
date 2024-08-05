import cv2
import numpy as np

def fiducial_focus_augmentation(image, landmarks):
    #print("Image dtype:", image.dtype)
    #print("Image shape:", image.shape)
    #print("Landmarks shape:", landmarks.shape)
    
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    
    if image.dtype == np.float32:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)
    
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    top_left = (10, 10)
    bottom_right = (50, 50)
    
    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError("Landmarks must be a 2D array with shape (num_landmarks, 2).")
    
    try:
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), thickness=cv2.FILLED)
    except cv2.error as e:
        print("OpenCV error:", e)
        raise
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image
