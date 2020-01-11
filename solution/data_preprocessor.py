import numpy as np


def load_data(path):
    """
        Loading data from file to 2 numpy arrays
        with attributes and labels and then shuffling them
    """
    with open(path) as file:
        lines = file.readlines()
        data = list(map(lambda line: line.split(), lines))
        np.random.shuffle(data)
        data = np.array(data)
        X = data[:, 0:-1]
        Y = data[:, -1]
        return X, Y
