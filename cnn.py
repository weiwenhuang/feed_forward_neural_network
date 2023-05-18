import numpy as np

def main():
    #choose,input_units,hidden_layers,hidden_units,output_units
    a = Neuron('sigmoid',2,1,2,1)
    test = np.array([2,3])
    a.feedforward(test)

def function(str,x):
    if str == 'Relu':
        return Relu(x)
    elif str == 'sigmoid':
        return sigmoid(x)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)


class Neuron:
    def __init__(self,choose,input_units,hidden_layers,hidden_units,output_units):
        self.input_units = input_units
        self.choose = choose
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.init_network()

    def init_network(self):
        self.layers_level = []
        # first level is input
        for i in range(self.hidden_layers):
            if i == 0:
                m = self.input_units
            else:
                m = len(self.layers_level[i - 1][1])
            n = self.hidden_units
            tem_weight = np.random.randn(m, n)
            tem_bias = np.random.randn(n, 1)
            self.layers_level.append([tem_weight, tem_bias])
        #predict
        tem_weight = np.random.randn(self.hidden_units, self.output_units)
        tem_bias = np.random.randn(self.output_units, 1)
        self.layers_level.append([tem_weight, tem_bias])
        #print(self.layers_level)
    
    def feedforward(self,x):

        #h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        #h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        #o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        self.forward_value = []
        self.forward_prevalue = []
        self.forward_value.append(x)
        for i in range(self.hidden_layers):
            arr1 = []
            arr2 = []
            for j in range(self.hidden_units):
                tem = 0
                for k in range(len(self.layers_level[i][0][j])):
                    tem += self.layers_level[i][0][j][k] * self.forward_value[i][k]
                tem += self.layers_level[i][1][j]
                prevalue = sigmoid(tem)
                arr1.append(tem)
                arr2.append(prevalue)
            self.forward_value.append(arr1)
            self.forward_prevalue.append(arr2)
            print(self.forward_value)
            print(self.forward_prevalue)

        return "sss"


if __name__ == "__main__":
    main()