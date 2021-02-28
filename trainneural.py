from makeneural import SimpleNeuralNetwork
import numpy as np

network = SimpleNeuralNetwork()
print("Network Weights")
print(network.weights)

'''
[EN] Data set for train neural network 
[PL] Zestaw danych do trenowania sieci neurnowej
'''
train_inputs = np.array(
    [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], ]
)

'''
[EN] Transpose the matrix vector 'train_input'
[PL] Transpozycja wektora macierzy 'train_input'
'''
train_outputs = np.array([[0, 1, 0, 0, 1, 0]]).T

'''
[EN] How many iteration for feed neural network
[PL] Ile iteracji(powtorzen) do nauki sieci neuronowej
'''
train_iterations = 50000

'''
[EN] Final learn a newly create network - input data, outputdata and the lastone how many iteration.
[PL] Finalnie uczymy nową sieć - dane wchodzące,wychodzące oraz ostatnie - ile iteracji.
'''
network.train(train_inputs, train_outputs, train_iterations)


print(network.weights)


print("Testing the data")
test_data = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], ])

for data in test_data:
    print(f"Result for {data} is:")
    print(network.propagation(data))