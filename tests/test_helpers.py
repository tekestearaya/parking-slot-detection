import numpy as np
import pytest
from src.utils.helpers import resize_image, normalize_image

def test_resize_image():
    dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)
    resized_image = resize_image(dummy_image)
    assert resized_image.shape == (640, 640, 3)

def test_normalize_image():
    dummy_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    normalized_image = normalize_image(dummy_image)
    assert normalized_image.max() <= 1.0
    assert normalized_image.min() >= 0.0
