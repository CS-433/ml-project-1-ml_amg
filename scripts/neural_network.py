import numpy as np
from helper import *

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    return x * (x > 0)

def relu_derivative(x):
    return 1 * (x > 0)

class NeuralNetwork:

    def __init__(self, x, y, layer_size, lr=0.1, n_epochs=10, batch_size=1, learning_rate_decay=0.95):
        self.input = x
        self.output = y
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.n_layers = len(self.layer_size)-1
        self.learning_rate_decay = learning_rate_decay

        self.weight = []
        self.bias = []

        for i in range(0, self.n_layers):
            self.weight.append(np.random.normal(size=(self.layer_size[i], self.layer_size[i+1])))
            self.bias.append(np.random.normal(size=(1, self.layer_size[i+1])))

    def forward(self, x):
        self.layers = []
        self.layers.append(x)
        for i in range(self.n_layers):
            z_output = np.dot(self.layers[i], self.weight[i]) + self.bias[i]
            if i < self.n_layers - 1:
                output = relu(z_output)
            else:
                output = sigmoid(z_output)
            self.layers.append(output)
        return self.layers

    # loss function
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def back_propagation(self, y):
        self.grad_list = []
        d_output = (self.layers[-1] - y) * sigmoid_derivative(self.layers[-1])
        self.grad_list.append(d_output)
        for i in range(self.n_layers-1, 0, -1):
            delta = np.dot(self.grad_list[-1], self.weight[i].T) * relu_derivative(self.layers[i])
            self.grad_list.append(delta)
        self.grad_list.reverse()

        for i in range(len(self.grad_list)):
            d_weight = np.dot(self.layers[i].T, self.grad_list[i])
            d_bias = self.grad_list[i]

            self.weight[i] -= d_weight * self.lr
            self.bias[i] -= d_bias * self.lr

        error = self.mse(self.layers[-1], y)
        return error

    def train(self, x_train, y_train):
        for epoch in range(self.n_epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                self.forward(x.reshape(1, len(x)))
                error += self.back_propagation(y.reshape(1, len(y)))
            if self.n_epochs % 10 == 0:
                # Decay learning rate
                self.lr *= self.learning_rate_decay
            error /= len(x_train)
            print(f'epoch{epoch} - error:{error}')

    def predict(self, input):
        output_list = self.forward(input)
        return output_list[-1]

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0