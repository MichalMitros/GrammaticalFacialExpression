from solution.file_preprocessor import consolidate_files, delete_unlabeled_records
from solution.data_preprocessor import load_data
from solution.neural_network import NeuralNetwork

'''
# Let's do it every single time - why not?
consolidate_files()
data, decisions = load_data('../preprocessed_data/datafile_v2.txt')
print(len(data))
delete_unlabeled_records()

# Load data and create neural network
data, decisions = load_data('../preprocessed_data/datafile_v2.txt')
nn = NeuralNetwork(len(data[0]), 3*len(data[0]), 18)
target_array = [0] * 18
error = 1
epoch = 0

while error >= 0.008:
    error = -1
    for row_index, row in enumerate(data):
        target_array[int(decisions[row_index])] = 1.
        err = nn.train(row, target_array)
        if err > error:
            error = err
        target_array[int(decisions[row_index])] = 0.
    epoch = epoch + 1
    print("Epoch: {}\tError: {}".format(epoch, error))

print("Finish")
'''

from math import floor
from solution.file_preprocessor import consolidate_files, delete_unlabeled_records
from solution.data_preprocessor import load_data, split_data
from solution.neural_network import NeuralNetwork


# Get class index from result
def get_max_index(array):
    index = 0
    for i in range(len(array)):
        if array[i] > array[index]:
            index = i
    return index


# Let's do it every single time - why not?
print('Preprocessing data...')
consolidate_files()
delete_unlabeled_records()

# Load data and create neural network
print('Loading data...')
data = load_data('../preprocessed_data/datafile_v2.txt')
nn = NeuralNetwork(len(data[0][0]), 3*len(data[0][0]), 18)
target_array = [0] * 18
accuracy = 0
epoch = 0

print('Learning started...')
while accuracy <= 0.9:
    accuracy = 0.
    epoch = epoch + 1

    # Train:
    for row_index, row in enumerate(data[0]):
        target_array[int(data[1][row_index])] = 1.
        nn.train(row, target_array)
        target_array[int(data[1][row_index])] = 0.

    # Test:
    for row_index, row in enumerate(data[0]):
        result = nn.feed_forward(row)
        target_array[int(data[1][row_index])] = 1.
        if get_max_index(result) == get_max_index(target_array):
            accuracy = accuracy + 1.
        target_array[int(data[1][row_index])] = 0.
    accuracy = accuracy / len(data[1])

    print('Epoch: {}\tAccuracy: {}%'.format(epoch, floor(accuracy*100)))

print('Learning finished')



