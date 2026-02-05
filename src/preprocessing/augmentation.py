import numpy as np

def horizontal_flip(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    return np.fliplr(image)
