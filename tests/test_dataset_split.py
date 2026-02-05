from src.dataset.split import split_dataset

def test_dataset_split_sizes():
    data = list(range(100))
    train, val, test = split_dataset(data)
    assert len(train) > len(val)
    assert len(test) > 0
