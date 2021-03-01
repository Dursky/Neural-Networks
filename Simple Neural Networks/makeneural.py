import numpy as np


class SimpleNeuralNetwork:
    """
    [EN]
    Simple neural network that checks if a given binary representation of a positive number is even

    [PL]
    Prosta sieć neuronowa, która sprawdza, czy dana binarna reprezentacja liczby dodatniej jest parzysta
    """

    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        """
        [EN]
        Sigmmoid function - smooth function that maps any number to a number from 0 to 1

        [PL]
        Funkcja Sigmoid - funkcja wygładzająca, która mapuje dowolną liczbę na liczbę od 0 do 1
        """
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        """
        [EN]
        Derivative of sigmoid function

        [PL]
        Pochodna funkcji sigmoidalnej
        """
        return x * (1 - x)

    def train(self, train_input, train_output, train_iters):
        for _ in range(train_iters):
            propagation_result = self.propagation(train_input)
            self.backward_propagation(
                propagation_result, train_input, train_output)

    def propagation(self, inputs):
        """
        [EN]
        Propagation process
        
        [PL]
        Proces propagowania (mnozenia)
        """
        return self.sigmoid(np.dot(inputs.astype(float), self.weights))

    def backward_propagation(self, propagation_result, train_input, train_output):
        """
        [EN]
        Backward propagation process 

        [PL]
        Wsteczny proces mnozenia
        """
        error = train_output - propagation_result
        self.weights += np.dot(
            train_input.T, error * self.d_sigmoid(propagation_result)
        )