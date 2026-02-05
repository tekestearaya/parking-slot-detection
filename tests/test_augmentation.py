import numpy as np
import pytest
from src.preprocessing.augmentation import horizontal_flip

def test_horizontal_flip():
    image = np.array([[1, 2], [3, 4]])
    flipped = horizontal_flip(image)
    assert (flipped == [[2, 1], [4, 3]]).all()
