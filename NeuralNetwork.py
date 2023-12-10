import numpy as np
from numpy import exp, random
import math

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(1)

class Neuron:

	def __init__(self, weights, bias):

		self.weights = weights
		self.bias = bias

	def sigmoid(self, x):

		output = 1/(1+exp(-x))

		return output

	def compute(self, inputs):

		self.output = self.sigmoid(np.dot(inputs, self.weights) + self.bias)

		return self.output

class Layer: 

	def __init__(self, numberOfNeurons, numberOfInputs):

		self.neurons = []
		self.outputs = []
		self.numberOfNeurons = numberOfNeurons
		self.numberOfInputs = numberOfInputs

		self.initialiseWeightsAndBiases()

		for i in range(0,numberOfNeurons):

			self.neurons.append(Neuron(self.weights, self.biases))

	def initialiseWeightsAndBiases(self):

		self.weights = 2 * random.random((self.numberOfInputs, self.numberOfNeurons)) - 1

		self.biases = 2 * random.random((1, self.numberOfNeurons)) - 1

	
	def forward(self, inputs):

		self.outputs = np.array([])

		for i in self.neurons:

			self.outputs = np.append(self.outputs, i.compute(inputs))

class NeuralNetwork:

	def __init__(self, layers):

		self.layers = layers

	def forwardPass(self, inputs):

		for i in range(0,len(layers)):

			if i == 0:

				layers[i].forward(inputs)	

			else:
				
				layers[i].forward(layers[i-1].outputs)

		return layers[-1].outputs

	def calculateError(self, predictedOutputs, trueOutputs):

		error = (trueOutputs - predictedOutputs) * predictedOutputs * (1 - predictedOutputs)

		return error

	def trainNetwork(self, trainingDataInputs, trainingDataOutputs, numberOfIterations):

		#initialise the best weights with random values

		for y in range(0, numberOfIterations):

			predictedOutputs = self.forwardPass(trainingDataInputs)

			error = self.calculateError(predictedOutputs, trainingDataOutputs)

			for i in layers[0].neurons:				

				i.weights += np.dot(trainingDataInputs.T, error.T)


	def visualiseNetwork(self):

		pass


#Layer(numberOfNeurons, numberOfInputs)

inputLayer = Layer( 1, 3)

layers = [inputLayer]

network1 = NeuralNetwork(layers)

inputTrainingData = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputTrainingData = [[0, 1, 1, 0]]

network1.trainNetwork(inputTrainingData, outputTrainingData, 10000)

outputs = network1.forwardPass([[0,1,1]])

print(outputs)




