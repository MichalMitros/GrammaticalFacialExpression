import unittest
import random
from solution.neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):

    def test_nn_feed_forward(self):
        nn = NeuralNetwork(2, 3, 1, learning_rate=0.15, random_seed=1)
        data = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.assertEqual(round(nn.feed_forward(data[0])[0], 7), 0.2348487)
        self.assertEqual(round(nn.feed_forward(data[1])[0], 7), 0.235499)
        self.assertEqual(round(nn.feed_forward(data[2])[0], 7), 0.2590947)
        self.assertEqual(round(nn.feed_forward(data[3])[0], 7), 0.2567514)

    def test_nn_training(self):
        nn = NeuralNetwork(2, 3, 1, learning_rate=0.15, random_seed=1)
        random.seed(1)
        data = [
            [0, 0, 0],
            [0 , 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        for i in range(6000):
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
        self.assertEqual(round(nn.feed_forward(data[0][:2])[0], 7), 0.0369205)
        self.assertEqual(round(nn.feed_forward(data[1][:2])[0], 7), 0.9679596)
        self.assertEqual(round(nn.feed_forward(data[2][:2])[0], 7), 0.9679559)
        self.assertEqual(round(nn.feed_forward(data[3][:2])[0], 7), 0.0270922)


if __name__ == '__main__':
    unittest.main()
