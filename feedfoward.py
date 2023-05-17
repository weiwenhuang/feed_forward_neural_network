import board as bd
import numpy as np



def main():
    board = [
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]]
    test = np.array([1,2,3])
    a = Neuron(3,'sigmoid',3,3,3)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

class Neuron:
    def __init__(self,input_units,choose,hidden_layers,hidden_units,output_units):
        self.input_units = input_units
        self.choose = choose
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.init_network()

    def feedforward(self,input_units,choose,hidden_layers,hidden_units,output_units):
        t = np.dot(self.weight, input_units) + self.bias
        if choose == 'sigmoid':
            return sigmoid(t)
        elif choose == 'Relu':
            return Relu(t)
        else:
            return False
        
    def init_network(self):
        self.network_layers = []
        #inition the layers
        for i in range(self.hidden_layers):
            if i == 0:
                m = self.input_units
            else:
                if type(self.network_layers[i - 1]) is list:
                    m = self.network_layers[i - 1][0].shape[1]
                else:
                    m = self.network_layers[i - 1].shape[1]
            n = self.hidden_units

            tem_weight = np.random.randn(m, n) * 0.01
            tem_bias = np.random.randn(n, 1) * 0.01
            self.network_layers.append([tem_weight, tem_bias])

        tem_weight = np.random.randn(self.hidden_units, self.output_units) * 0.01
        tem_bias = np.zeros((self.output_units, 1))
        self.network_layers.append([tem_weight, tem_bias])




if __name__ == "__main__":
    main()