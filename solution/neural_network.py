import numpy as np


def sigmoid(x, deriv=False):
    """
    Sigmoid function
    :param x: argument of function
    :param deriv: boolean if result should be derivative of sigmoid (False by default)
    :return: Result of sigmoid function or result of sigmoid function's derivative

    """

    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, inputs, hidden1, hidden2, outputs, learning_rate=0.05, activation=sigmoid, random_seed=None):
        """
        Constructs neural network with random weights and biases (values from -1 to 1)
        :param inputs: number of inputs of neural network
        :param hidden1: number of neurons in first hidden layer
        :param hidden2: number of neurons in second hidden layer
        :param outputs: number of outputs (neurons in output layer)
        :param learning_rate: learning rate of neural network (0.05 by default)
        :param activation: global activation function for this neural network (sigmoid by default)
        :param random_seed: seed for generating random weights and biases for debug/testing purposes (None by default)

        """

        if random_seed:
            np.random.seed(random_seed)
        # set network parameters
        self.inputs = inputs
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.activation = activation
        # initialize weights ( matrices with values between -1 and 1)
        self.hidden1_weights = 2 * np.random.random((hidden1, inputs)) - 1
        self.hidden2_weights = 2 * np.random.random((hidden2, hidden1)) - 1
        self.output_weights = 2 * np.random.random((outputs, hidden2)) - 1
        # initialize biases (arrays with values between -1 and 1)
        self.hidden1_biases = 2 * np.random.rand(hidden1) - 1
        self.hidden2_biases = 2 * np.random.rand(hidden2) - 1
        self.output_biases = 2 * np.random.rand(outputs) - 1

    def feed_forward(self, input_array):
        """
        Getting result for input from the neural network
        :param input_array: array of inputs to get the results from
        :return: array of outputs from neural network

        """

        # Calculate hidden layer's output
        hidden = np.dot(self.hidden1_weights, input_array)
        hidden = hidden + self.hidden1_biases
        hidden = self.activation(hidden)

        hidden = np.dot(self.hidden2_weights, hidden)
        hidden = hidden + self.hidden2_biases
        hidden = self.activation(hidden)

        # Calculate output's layer output (final result)
        output = np.dot(self.output_weights, hidden)
        output = output + self.output_biases
        output = self.activation(output)

        # return final result (array)
        return output

    #   Choo choo!
    def train(self, input_array, target_array):
        """
        Executing single training step in the neural network
        :param input_array: array of inputs used o train neural network
        :param target_array: array of expected, correct answers
        :return: MSE (Mean-Square-Error) for this input before training step (Float)

        """

        # FEED FORWARD ALGORITHM:
        # Calculate hidden layer's output
        hidden1 = np.dot(self.hidden1_weights, input_array)
        hidden1 = hidden1 + self.hidden1_biases
        hidden1 = self.activation(hidden1)

        hidden2 = np.dot(self.hidden2_weights, hidden1)
        hidden2 = hidden2 + self.hidden2_biases
        hidden2 = self.activation(hidden2)

        # Calculate output's layer output (final result)
        output = np.dot(self.output_weights, hidden2)
        output = output + self.output_biases
        output = self.activation(output)

        # ERRORS CALCULATING
        output_errors = target_array - output
        hidden2_errors = self.output_weights.T.dot(output_errors)
        hidden1_errors = self.hidden2_weights.T.dot(hidden2_errors)

        # BACKPROPAGATION
        # Calculate the output layer's gradients
        output_gradients = self.activation(output, deriv=True)
        output_gradients = np.multiply(output_gradients, output_errors)
        output_gradients = np.multiply(output_gradients, self.learning_rate)

        # Calculate the output layer's deltas
        output_deltas = np.dot(np.array([output_gradients]).T, np.array([hidden2]))

        # Adjusting output layer's weights and biases
        self.output_weights = self.output_weights + output_deltas
        self.output_biases = self.output_biases + output_gradients

        # Calculate second hidden layer's gradients
        hidden2_gradients = self.activation(hidden2, deriv=True)
        hidden2_gradients = np.multiply(hidden2_gradients, hidden2_errors)
        hidden2_gradients = np.multiply(hidden2_gradients, self.learning_rate)

        # Calculate second hidden layer's deltas
        hidden2_deltas = np.dot(np.array([hidden2_gradients]).T, np.array([hidden1]))

        # Adjusting second hidden layer's weights and biases
        self.hidden2_weights = self.hidden2_weights + hidden2_deltas
        self.hidden2_biases = self.hidden2_biases + hidden2_gradients

        # Calculate first hidden layer's gradients
        hidden1_gradients = self.activation(hidden1, deriv=True)
        hidden1_gradients = np.multiply(hidden1_gradients, hidden1_errors)
        hidden1_gradients = np.multiply(hidden1_gradients, self.learning_rate)

        # Calculate first hidden layer's deltas
        hidden1_deltas = np.dot(np.array([hidden1_gradients]).T, np.array([input_array]))

        # Adjusting first hidden layer's weights and biases
        self.hidden1_weights = self.hidden1_weights + hidden1_deltas
        self.hidden1_biases = self.hidden1_biases + hidden1_gradients

        return (1./self.inputs)*np.sum(np.square(output_errors))
