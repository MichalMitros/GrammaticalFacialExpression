# NEURAL NETWORK TEST (XOR example)
from solution.neural_network import NeuralNetwork
from random import shuffle

# Training data
# [x1, x2, y]
# x1, x2 - attributes
# y - decision
data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# Creating neural network with custom learning rate
nn = NeuralNetwork(2, 3, 1, learning_rate=0.15)

# Training
for i in range(6000):
    shuffle(data)
    nn.train(data[0][:2], data[0][2:])
    nn.train(data[1][:2], data[1][2:])
    nn.train(data[2][:2], data[2][2:])
    nn.train(data[3][:2], data[3][2:])

# Restore original data (previous were shuffled)
data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# Get results
for i in range(0, len(data)):
    result = nn.feed_forward(data[i][:2])
    print('Got: {}\tExpected: {}\tError: {}'.format(
        result,
        data[i][2:],
        abs(data[i][2:]-result))
    )
