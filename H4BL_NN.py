import numpy as np

class NeuralNetwork():
    def __init__(self):
        n_inputs = 4
        n_hidden = 5
        n_outputs = 2

        np.random.seed(1)
        self.synaptic_weights0 = 2 * np.random.random((n_inputs, n_hidden)) - 1
        self.synaptic_weights1 = 2 * np.random.random((n_hidden, n_outputs)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):

            l0 = training_inputs
            l1 = self.sigmoid(np.dot(l0, self.synaptic_weights0))
            l2 = self.sigmoid(np.dot(l1, self.synaptic_weights1))

            l2_error = training_outputs - l2 #gucken wie mit array umgegangen wird
            if (iteration % 20000) == 0:
                print("Error: " + str(np.mean(np.abs(l2_error))))

            l2_delta = l2_error*self.sigmoid_derivative(l2)
            l1_error = l2_delta.dot(self.synaptic_weights1.T)
            l1_delta = l1_error * self.sigmoid_derivative(l1)
            self.synaptic_weights1 += l1.T.dot(l2_delta)
            self.synaptic_weights0 += l0.T.dot(l1_delta)
        print("Output after Training: ")
        print(l2)

    def think(self, inputs):
        inputs = inputs.astype(float)

        l0 = inputs
        l1 = self.sigmoid(np.dot(l0, self.synaptic_weights0))
        l2 = self.sigmoid(np.dot(l1, self.synaptic_weights1))

        return l2

