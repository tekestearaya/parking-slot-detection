# tests/test_training.py

import pytest
from src.model.train import train_model

def test_train_model_success():
    assert train_model([1, 2, 3]) is True

def test_train_model_empty():
    with pytest.raises(ValueError):
        train_model([])
