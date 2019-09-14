from numpy import exp, array, random, dot, log
from tqdm import trange

class Activation:
	def activation(x):
		pass
	
	def derivative(x):
		pass

class Sigmoid(Activation):
	
	def activation(x):
		return 1 / (1 + exp(-x))
	
	def derivative(x):
		return x * (1 - x)

class TanH(Activation):

	def activation(x):
		return (2 / (1 + exp(-2 * x))) - 1
	
	def derivative(x):
		return 1 - pow(self.activation(x), 2)

class SoftPlus(Activation):

	def activation(x):
		return log(1 + exp(x))
	
	def derivative(x):
		return 1 / (1 + exp(-x))

class Layer():
    def __init__(self, input_count, neuron_count, activation = Sigmoid):
        
        self.activation = activation
        self.weights = 2 * random.random((input_count, neuron_count)) - 1
        self.biases = 2 * random.random((1, neuron_count)) - 1
        self.outputs = []
        self.inputs = []
        self.error = []
        self.delta = []

    def calc_error(self, layer_before):
        self.error = layer_before.delta.dot(layer_before.weights.T)

    def calc_delta(self):
        self.delta = self.error * self.activation.derivative(self.outputs)

    def adjust(self, learning_rate):
        self.weights += self.inputs.T.dot(self.delta) * learning_rate
        self.biases += self.delta[0] * learning_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation.activation(dot(inputs, self.weights) + self.biases)


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def train(self, X, y, epochs):
        y = y.T

        # training epochs
        for iteration in trange(epochs):

            #Forward Pass
            self.forward(X)

            # calculate error of output layer based on difference to desired output
            self.layers[-1].error = y - self.layers[-1].outputs
            self.layers[-1].calc_delta()
            
            # itterate backwards over all not handled layers
            for i in range(len(self.layers) - 2, -1, -1):
                # calculate the error value based on the layer below
                self.layers[i].calc_error(self.layers[i + 1])
                # calculate the delta value
                self.layers[i].calc_delta()
            
            # itterate forwards over all layers
            for l in self.layers:
                # adjust weights based on delta
                l.adjust(0.1)

    def forward(self, inputs):

        # forward the first layer with the input values
        self.layers[0].forward(inputs)

        #itterate over all layers exept the first one
        for i in range(1, len(self.layers)):
            # forward with outputs of the layer bevore as input
            self.layers[i].forward(self.layers[i - 1].outputs)
        
        # return the predictions
        return self.layers[-1].outputs

if __name__ == "__main__":

    random.seed(1)
    
    layers = [ Layer(3, 4), Layer(4, 4), Layer(4, 4), Layer(4, 1)]

    NN = NeuralNetwork(layers)

    X = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    y = array([[0, 1, 1, 1, 1, 0, 0]])

    NN.train(X, y, 60000)

    print( "Considering a new situation [1, 1, 0] -> : " + str(NN.forward(array([1, 1, 0]))) )
