from numpy import exp, array, random, dot
from tqdm import trange

def sigmoid(x):
    return 1 / (1 + exp(-x))
    
def sigmoid_deriv(x):
    return x * (1 - x)

class NeuronLayer():
    def __init__(self, neuron_count, input_count):

        self.weights = 2 * random.random((input_count, neuron_count)) - 1
        self.outputs = []
        self.inputs = []
        self.error = []
        self.delta = []

    def calc_error(self, layer_before):
        self.error = layer_before.delta.dot(layer_before.weights.T)

    def calc_delta(self):
        self.delta = self.error * sigmoid_deriv(self.outputs)

    def adjust(self):
        self.weights += self.inputs.T.dot(self.delta)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = sigmoid(dot(inputs, self.weights))


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
                l.adjust()

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
    
    layers = [ NeuronLayer(4, 3), NeuronLayer(4, 4), NeuronLayer(4, 4), NeuronLayer(1, 4)]

    NN = NeuralNetwork(layers)

    X = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    y = array([[0, 1, 1, 1, 1, 0, 0]])

    NN.train(X, y, 60000)

    print( "Considering a new situation [1, 1, 0] -> : " + str(NN.forward(array([1, 1, 0]))) )
