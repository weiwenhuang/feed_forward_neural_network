import board as bd
import numpy as np



def main():
    board = [
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]]
    test = np.array([1,2,3])
    a = Neuron(3,'sigmoid',3,3,3)
    a.classify(test)
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
        
    # part3   
    def init_network(self):
        self.layers_level = []
        #inition the layers
        for i in range(self.hidden_layers):
            if i == 0:
                m = self.input_units
            else:
                if type(self.layers_level[i - 1]) is list:
                    m = self.layers_level[i - 1][0].shape[1]
                else:
                    m = self.layers_level[i - 1].shape[1]
            n = self.hidden_units

            tem_weight = np.random.randn(m, n) * 0.01
            tem_bias = np.random.randn(n, 1) * 0.01
            self.layers_level.append([tem_weight, tem_bias])

        tem_weight = np.random.randn(self.hidden_units, self.output_units) * 0.01
        tem_bias = np.zeros((self.output_units, 1))
        self.layers_level.append([tem_weight, tem_bias])
    
    def classify(self,train_data):
        self.train_data = train_data
        self.forward_value = []
        self.forward_outvalue = []

        # input layer to hidden layer
        weight_0 = self.layers_level[0][0]
        bias_0 = self.layers_level[0][1]
        inputdata = train_data

        value = np.dot(weight_0, inputdata) + bias_0
        outvalue = sigmoid(value)
        self.forward_value.append(value)
        self.forward_outvalue.append(outvalue)

        # hidden layer to outputlayer
        for i in range(1,len(self.layers_level)):
            weight_i = self.layers_level[i][0]
            bias_i = self.layers_level[i][1]
            data = self.forward_outvalue[i-1]
            value = np.dot(weight_i, data) + bias_i
            outvalue = sigmoid(value)
            self.forward_value.append(value)
            self.forward_outvalue.append(outvalue)
        print(self.forward_outvalue)
        




if __name__ == "__main__":
    main()