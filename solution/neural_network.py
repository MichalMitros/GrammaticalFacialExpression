class NeuralNetwork:

    def __init__(self, num_of_inputs, num_of_hidden, num_of_outputs):
        self.num_of_inputs = num_of_inputs
        self.num_of_hidden = num_of_hidden
        self.num_of_outputs = num_of_outputs

    def print(self):
        print("I am Neural Network!")