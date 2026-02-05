from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2, val_size=0.1):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    train, val = train_test_split(train, test_size=val_size, random_state=42)
    return train, val, test
