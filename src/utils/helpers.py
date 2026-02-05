import cv2
import numpy as np

def resize_image(image, size=(640, 640)):
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    return cv2.resize(image, size)

def normalize_image(image):
    return image.astype("float32") / 255.0
