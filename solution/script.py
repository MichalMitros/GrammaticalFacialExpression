from solution.file_preprocessor import consolidate_files, delete_unlabeled_records
from solution.data_preprocessor import load_data
from solution.neural_network import NeuralNetwork


# Let's do it every single time - why not?
consolidate_files()
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

