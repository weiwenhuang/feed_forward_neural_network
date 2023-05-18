import numpy as np

def main():
    #choose,input_units,hidden_layers,hidden_units,output_units
    a = Neuron('sigmoid',2,1,2,2)


    data = np.array([
    [-2, -1],  # Alice
    [25, 6],   # Bob
    [17, 4],   # Charlie
    [-15, -6], # Diana
    ])

    all_y_trues = np.array([
    [0,1], # Alice
    [1,0], # Bob
    [1,0], # Charlie
    [0,1], # Diana
    ])

    a.update_weight(data,all_y_trues)
    print(a.feedforward(np.array([25,6])))

def function(str,x):
    if str == 'Relu':
        return Relu(x)
    elif str == 'sigmoid':
        return sigmoid(x)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()



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
            tem = 0
            for i in range(len(self.forward_prevalue[-1])):
                tem += self.layers_level[-1][0][i] * self.forward_prevalue[-1][i]
            tem += self.layers_level[-1][1][0][0]
            predict = sigmoid(tem)
            self.forward_value.append(tem)
            self.forward_prevalue.append(predict)
        return predict
    
    def update_weight(self,data, ally_true):
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset
        for epoch in range(epochs):
            for x, y_true in zip(data, ally_true):
                predicty = self.feedforward(x)
                d_L_d_ypred = -2 * (y_true - predicty)
                for i in range(1,len(self.forward_prevalue)+1):
                    t = i+1
                    for j in range(len(self.layers_level[-i][0])):
                        for k in range(len(self.layers_level[-i][0][j])):
                            if t == len(self.forward_prevalue)+1:
                                h = x
                            else:
                                h = self.forward_prevalue[-t]
                            #self.layers_level[-i][0][j][k] -= learn_rate * d_L_d_ypred * h[k] * deriv_sigmoid(self.forward_prevalue[-i][k])
                            for index in range(len(self.layers_level[-i][1])):
                                self.layers_level[-i][0][j][k] -= learn_rate * d_L_d_ypred[index] * h[k] * deriv_sigmoid(self.forward_prevalue[-i][k])
                                self.layers_level[-i][1][index] -= learn_rate * d_L_d_ypred[index] * deriv_sigmoid(np.float64(self.forward_prevalue[-i][k]))
                                #learn_rate * d_L_d_ypred * deriv_sigmoid(np.float64(self.forward_prevalue[-i][k]))
                            #self.layers_level[-i][1] -= learn_rate * d_L_d_ypred * deriv_sigmoid(self.forward_prevalue[-i][k])
                    

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(ally_true, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


if __name__ == "__main__":
    main()