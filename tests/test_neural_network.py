import unittest
import random
from solution.neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):

    def test_nn_feed_forward(self):
        nn = NeuralNetwork(2, 3, 2, 1, learning_rate=0.15, random_seed=1)
        data = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.assertIsNotNone(nn.feed_forward(data[0][:2])[0])
        self.assertIsNotNone(nn.feed_forward(data[1][:2])[0])
        self.assertIsNotNone(nn.feed_forward(data[2][:2])[0])
        self.assertIsNotNone(nn.feed_forward(data[3][:2])[0])

    def test_nn_training(self):
        nn = NeuralNetwork(2, 3, 2, 1, learning_rate=0.15, random_seed=1)
        random.seed(1)
        data = [
            [0, 0, 0],
            [0 , 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        for i in range(2000):
            random.shuffle(data)
            nn.train(data[0][:2], data[0][2:])
            nn.train(data[1][:2], data[1][2:])
            nn.train(data[2][:2], data[2][2:])
            nn.train(data[3][:2], data[3][2:])
        data = [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self.assertLess(nn.feed_forward(data[0][:2])[0], .1)
        self.assertGreater(nn.feed_forward(data[1][:2])[0], .9)
        self.assertGreater(nn.feed_forward(data[2][:2])[0], .9)
        self.assertLess(nn.feed_forward(data[3][:2])[0], .1)


if __name__ == '__main__':
    unittest.main()
