import numpy as np
import pytest
from src.preprocessing.preprocess import preprocess_image

def test_preprocess_image_shape():
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    output = preprocess_image(image)
    assert output.shape == (640, 640, 3)

def test_preprocess_normalization():
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    output = preprocess_image(image)
    assert output.max() <= 1.0
