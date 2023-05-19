import numpy as np
import data as da
import board as bd

np.set_printoptions(suppress=True)
def main():
    #choose,input_units,hidden_layers,hidden_units,output_units
    #defhexapawntest()
    #defhexapawntest()
    testweight()


def defhexapawntest():
    a = Neuron('sigmoid',10,2,2,1)
    all_y_trues = np.array([])
    tran_data = []
    for i in da.boards :
        all_y_trues =  np.append(all_y_trues,bd.minmax(i,1))
        tran_data.append(toarr(i))
    a.update_weight(tran_data,all_y_trues)
    print(a.feedforward([1, -1, -1, 0, 0, 0, -1, 1, 1, 1]))



def toarr(x):
    res = [1]
    for i in range(len(x)):
        for j in range(len(x[i])):
            res.append(x[i][j])
    return res

def testweight():
    #choose,input_units,hidden_layers,hidden_units,output_units
    a = Neuron('sigmoid',2,2,3,1)
    data = np.array([
    [-2, -1],  
    [25, 6],   
    [17, 4],   
    [-15, -6], 
    ])
    all_y_trues = np.array([
    1, 
    0, 
    0, 
    1, 
    ])

    emily = np.array([-7, -3]) # 128 pounds, 63 inches
    #frank = np.array([20, 2])  # 155 pounds, 68 inches
    print("Emily: %.3f" % a.feedforward(emily)) # 0.951 - F
    #print("Frank: %.3f" % a.feedforward(frank)) # 0.039 - M

def addtest():

    a = Neuron('sigmoid',2,2,2,2)


    data = np.array([
    [0, 0],  
    [0, 1],   
    [1, 0],   
    [1, 1], 
    ])

    all_y_trues = np.array([
    [0,0], 
    [0,1], 
    [0,1], 
    [1,0], 
    ])
    a.update_weight(data,all_y_trues)
    print(a.feedforward(np.array([1,1])))


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
            tem_weight = np.random.randn(n, m)
            tem_bias = np.random.randn(n, 1)
            self.layers_level.append([tem_weight, tem_bias])
        #predict
        tem_weight = np.random.randn(self.hidden_units, self.output_units)
        tem_bias = np.random.randn(self.output_units, 1)
        self.layers_level.append([tem_weight, tem_bias])
        print("------NETWAREK LAYERS-------")
        print(self.layers_level)
        print("------------------")
    
    def feedforward(self,x):

        #h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        #h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        #o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        self.forward_sum_value = []
        self.forward_pre_value = []
        self.forward_sum_value.append(x)
        self.forward_pre_value.append(x)
        #first hidden layer
        arr1 = []
        arr2 = []
        sum_value = 0
        for j in range(self.hidden_units):
            sum_value = 0
            for i in range(len(x)):
               #w*x 
                sum_value += x[i] * self.layers_level[0][0][j][i]
            sum_value += self.layers_level[0][1][j]

            predict = sigmoid(sum_value)
            arr1.append(sum_value)
            arr2.append(predict)
        self.forward_sum_value.append(arr1)
        self.forward_pre_value.append(arr2)

        #rest
        for i in range(1,self.hidden_layers):
            arr1 = []
            arr2 = []
            for j in range(self.hidden_units):#for rach unit
                sum_value = 0
                print()
                for k in range(len(self.layers_level[i][0][j])):
                    # weight * prevalue
                    sum_value += self.layers_level[i][0][j][k] * self.forward_pre_value[i][k]
                    #print('x:',self.layers_level[i][0][j][k],'weight:',self.forward_pre_value[i][k])
                #print('bias:',self.layers_level[i][1][j])
                #plus bias
                sum_value += self.layers_level[i][1][j]
                arr1.append(sum_value)
                #get predict value fron function
                predict = sigmoid(sum_value)
                arr2.append(predict)
            self.forward_sum_value.append(arr1)
            self.forward_pre_value.append(arr2)

        #outputlayer:
        arr1 = []
        arr2 = []
        for i in range(self.output_units):
            sum_value = 0
            for j in range(self.hidden_units):
                #print('x:',self.forward_pre_value[-1][j],'weight:',self.layers_level[-1][0][j])
                sum_value += self.forward_pre_value[-1][j] * self.layers_level[-1][0][j]
            sum_value += self.layers_level[-1][1][i]
            #print('bias:',self.layers_level[-1][1][i])
            arr1.append(sum_value)
            #get predict value fron function
            predict = sigmoid(sum_value)
            arr2.append(predict)
        self.forward_sum_value.append(arr1)
        self.forward_pre_value.append(arr2)
        return predict
    
    '''
    def update_weight(self,data, ally_true):
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset
        for epoch in range(epochs):
            for x, y_true in zip(data, ally_true):
                predicty = self.feedforward(x)
                d_L_d_ypred = -2 * (y_true - predicty)
                
                

                    

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(ally_true, y_preds)
                print("Epoch %d learning rate: %.3f" % (epoch, loss))'''


if __name__ == "__main__":
    main()