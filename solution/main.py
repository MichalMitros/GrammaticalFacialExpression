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
data = split_data(data, 0.8)
nn = NeuralNetwork(len(data[0][0]), 150, 100, 9, learning_rate=0.01, random_seed=1)
target_array = [0] * 9

accuracy = 0
prev_accuracy = 0
epoch = 0
max_epoch = 100
min_accuracy = 0.9

print('Learning started...')
while accuracy <= min_accuracy and epoch < max_epoch:
    prev_accuracy = floor(accuracy*10000)
    accuracy = 0.
    epoch = epoch + 1

    # Train:
    for row_index, row in enumerate(data[0]):
        target_array[int(data[1][row_index])] = 1.
        nn.train(row, target_array)
        target_array[int(data[1][row_index])] = 0.

    # Test:
    for row_index, row in enumerate(data[2]):
        result = nn.feed_forward(row)
        target_array[int(data[3][row_index])] = 1.
        if get_max_index(result) == get_max_index(target_array):
            accuracy = accuracy + 1.
        target_array[int(data[3][row_index])] = 0.
    accuracy = accuracy / len(data[3])

    if prev_accuracy >= floor(accuracy*10000) and nn.learning_rate > 0.001:
        nn.learning_rate = nn.learning_rate * 0.9

    print('Epoch: {}\tAccuracy: {}%'.format(epoch, floor(accuracy*100)))

if epoch is not max_epoch:
    print('Learning finished. Final accuracy: {}'.format(accuracy))
else:
    print('Learning finished. Target accuracy not reached!')
