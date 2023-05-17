import board as bd
import numpy as np
import random



def main():
    board = [
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]]
    test = np.array([1,2,3])
    print("\n\nADDER NETWORK")
    a = Neuron(2,'Relu',2,2,2)
    adder(a,100)



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

        value = np.matmul(weight_0, inputdata) + bias_0
        outvalue = function(self.choose,value)
        self.forward_value.append(value)
        self.forward_outvalue.append(outvalue)

        # hidden layer to outputlayer
        for i in range(1,len(self.layers_level)):
            weight_i = self.layers_level[i][0]
            bias_i = self.layers_level[i][1]
            data = self.forward_outvalue[i-1]
            value = weight_i @ data + bias_i
            outvalue = function(self.choose,value)
            self.forward_value.append(value)
            self.forward_outvalue.append(outvalue)

    def update_weights(self,train_ans):
        self.back_delta = []
        #l2 loss function
        l = np.power(train_ans - self.forward_outvalue[-1],2)
        dl = -2 * (train_ans - self.forward_outvalue[-1])
        for i in reversed(range(len(self.layers_level))):
            in_i = self.forward_value[i]
            gp_i = function(self.choose,in_i)
            if i == len(self.layers_level) - 1:
                d_i = dl * gp_i
            else:
                d_ip1 = self.back_delta[-1]  
                w_ip1 = self.layers_level[i + 1][0]  
                d_i = np.matmul(w_ip1,d_ip1) * gp_i
            self.back_delta.append(d_i)
        self.back_delta.reverse()
        for i,layer in enumerate(self.layers_level):
            w_i = layer[0]
            d_i = self.back_delta[i]
            # input to hidden transition
            if i == 0:
                x_i = self.train_data
                w_i += np.matmul(x_i,d_i.T)*0.01
            else:
                a_i = self.forward_value[i - 1]
                w_i += np.matmul(a_i,d_i.T)*0.01
            layer[0] = w_i
'''
def adder(feedfoward, tran_range):
    df = [(np.array([[0], [0]]), np.array([[0], [0]])),(np.array([[0], [1]]), np.array([[0], [1]])),(np.array([[1], [0]]), np.array([[0], [1]])),(np.array([[1], [1]]), np.array([[1], [0]])),]
    #tran
    for i in range(tran_range):
        d = random.choice(df)
        feedfoward.classify(d[0])
        feedfoward.update_weights(d[1])
    for j in df:
        feedfoward.classify(j[0])
        predict = feedfoward.forward_outvalue[-1]
        print('Input: ',j[0].flatten())
        print('Output: ',j[1].flatten())
        print('predict: ',predict.flatten())'''
def adder(net, epochs: int = 100):
    """Automate training and testing of Adder given a FeedForwardNet.

    Args:
        net (FeedForwardNet): Instance of FeedForwardNet
        epochs (int, optional): Number of training rounds. Defaults to 20.
    """
    # Input/Output Pairs from writeup
    data = [
        (np.array([[0], [1]]), np.array([[0], [1]])),
        (np.array([[0], [0]]), np.array([[0], [0]])),
        (np.array([[1], [0]]), np.array([[0], [1]])),
        (np.array([[1], [1]]), np.array([[1], [0]])),
    ]

    # Train
    for i in range(epochs):
        d = random.choice(data)
        net.classify(d[0])
        net.update_weights(d[1])

    # Test
    for pair in data:
        net.classify(pair[0])
        pred = net.forward_outvalue[-1]
        print(
            f"Input: {pair[0].flatten()}\n"
            f"Output: {pair[1].flatten()}\n"
            f"Predicted: {pred}\n"
        )   



if __name__ == "__main__":
    main()