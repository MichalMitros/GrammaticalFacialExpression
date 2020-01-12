import numpy as np


def normalize_data(data):
    """
    Min-max normalization
    """
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)


def load_data(path):
    """
        Loading data from file to 2 numpy arrays
        with attributes and labels and then shuffling them
    """
    with open(path) as file:
        lines = file.readlines()
        data = list(map(lambda line: line.split(), lines))
        np.random.shuffle(data)
        data = np.array(data, dtype=np.float)
        X = data[:, 0:-1]
        X = normalize_data(X)
        Y = data[:, -1]
        return X, Y


def split_data(data, train_set_size):
    """
        Splitting data into train records, train labels,
        test records and test labels numpy arrays
    """
    X = data[0]
    Y = data[1]
    X_train = X[0:int(len(X) * train_set_size), :]
    Y_train = Y[0:int(len(X) * train_set_size)]
    X_test = X[int(len(X) * train_set_size):, :]
    Y_test = Y[int(len(X) * train_set_size):]
    return X_train, Y_train, X_test, Y_test
