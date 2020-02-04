"""
Module for neural network training and execution.
"""


import random
import numpy as np

class Network():
    """neural network object."""

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        """computes the output of the network."""
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """stochastic gradient descent algorithm."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.mini_update_batch(mini_batch, eta)
            if test_data:
                print('epoch{}: {}/{}'.
                      format(i, self.evaluate(test_data), n_test))
            else:
                print('Epoch{} complete'.format(i))

    def mini_update_batch(self, mini_batch, eta):
        """modify the biases and weights"""
        nabla_b = [np.zeros(x.shape) for x in self.biases]
        nabla_w = [np.zeros(x.shape) for x in self.weights]
        inp = np.asarray([x.ravel() for x, y in mini_batch]).transpose()
        outp = np.asarray([y.ravel() for x, y in mini_batch]).transpose()
        nabla_b, nabla_w = self.backprop(inp, outp)
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """backpropagation algorithm."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = (self.cost_derivative(activations[-1], y) *
                 self.sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta.sum(1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """evaluate network on test data."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """derivative of the cost function."""
        return output_activations-y

    def sigmoid(self, z):
        """sigmoid activation function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """derivative of the sigmoid activation funtion."""
        return self.sigmoid(z)*(1-self.sigmoid(z))
