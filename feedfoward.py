# Author: Weiwen Huang, Yaxi Huang
# Course: CISC481

import numpy as np
import data as da
import board as bd

np.set_printoptions(suppress=True)
def main():
    #choose,input_units,hidden_layers,hidden_units,output_units
    hexapawntest() #in minmax it will return [-1,1] in feedfoward it will return [0,1] 0 instead -1
    #addtest()
    #testweight()

def hexapawntest():
#   for this test 2 hiiden layers and 21 hideen unit is good to predict the result, even more layer and more unit will make it better, but it will take too much time to run
#   it will sent current board to action funtion to get all posstible action and then use a for loop to find witch one have biggest predict vlaue
#   so it will print biggest value and map
#   tran data from data.py i didn't make too much tran data. more tran data we have, the result will more close to correct answer
#   Neuron(choose,input_units,hidden_layers,hidden_units,output_units)
    a = Neuron('sigmoid',10,2,2,1)
    all_y_trues = np.array([])
    tran_data = []
    maxscore = 0
    maxsboard = []
    for i in da.boards :
        all_y_trues =  np.append(all_y_trues,bd.minmax(i,1))
        tran_data.append(toarr(i))
    # (data, ally_true,learn_rate,tran_time)
    a.update_weight(tran_data,all_y_trues.reshape(len(all_y_trues),1),0.1,1000)
    #input data
    player = 1
    current_state =[[-1,0,-1],
        [1,-1,0],
        [0,1,1]]
    act_list = bd.ACTIONS(current_state,player)
    for i in act_list:
        arr_i = bd.toarr(i)
        new_state = bd.RESULT(current_state,[arr_i,act_list[i][0]])
        #convert to 1Dlist with payer
        res = [player]
        for i in range(len(new_state)):
            for j in range(len(new_state[0])):
                res.append(new_state[i][j])
        #use model to predict if better than current max predict keeep
        score = a.classify(res)
        if score[0] > maxscore:
            maxscore = score[0]
            maxsboard = new_state
    print('atfer model precited, the best next move for player:',player,'predict value:',maxscore,'is:\n',maxsboard)

def toarr(x):
    res = [1]
    for i in range(len(x)):
        for j in range(len(x[i])):
            res.append(x[i][j])
    return res

def testweight():
    #choose,input_units,hidden_layers,hidden_units,output_units
    a = Neuron('sigmoid',2,2,2,1)
    data = np.array([
    [-2, -1],  
    [25, 6],   
    [17, 4],   
    [-15, -6], 
    ])
    all_y_trues = np.array([
    [1], 
    [0], 
    [0], 
    [1], 
    ])
        # data, ally_true,learn_rate,epochs
    a.update_weight(data,all_y_trues,0.1,1000)
    emily = np.array([-7, -3]) # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print("Emily: ",a.classify(emily)) # 0.951 - F
    print("Frank:",a.classify(frank)) # 0.039 - M

def addtest():
    #choose,input_units,hidden_layers,hidden_units,output_units
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
        # data, ally_true,learn_rate,epochs
    a.update_weight(data,all_y_trues,0.1,1000)
    print('ans : ',a.classify(np.array([1,0])))

#this function is transfer to the function it use
def function(str,x):
    if str == 'Relu':
        return Relu(x)
    elif str == 'sigmoid':
        return sigmoid(x)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

def deriv_function(str,x):
    if str == 'Relu':
        return deriv_Relu(x)
    elif str == 'sigmoid':
        return deriv_sigmoid(x)

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def deriv_Relu(x):
# Derivative of relu: f'(x) = f(x) * (1 - f(x))
  fx = Relu(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()
# this is l2 loss function
def L2_loss(y_true,y_pre):
    return np.sum(np.square(y_true-y_pre))

class Neuron:
    def __init__(self,choose,input_units,hidden_layers,hidden_units,output_units):
        self.input_units = input_units
        self.choose = choose
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.init_network()
    #part3
    def init_network(self):
        self.layers_level = []
        # first level is input

        for i in range(self.hidden_layers):
            #input layers
            if i == 0:
                m = self.input_units
            #hidden layers
            else:
                m = len(self.layers_level[i - 1][1])
            n = self.hidden_units
            tem_weight = np.random.randn(n, m)
            tem_bias = np.random.randn(n, 1)
            self.layers_level.append([tem_weight, tem_bias])
        #output layers
        tem_weight = np.random.randn(self.output_units, self.hidden_units)
        tem_bias = np.random.randn(self.output_units, 1)
        self.layers_level.append([tem_weight, tem_bias])
    
    def classify(self,x):
        self.forward_sum_value = []#list save the sum of each unit
        self.forward_pre_value = []#list save the predict value for each unit
        self.forward_sum_value.append(x)# the first is input value 
        self.forward_pre_value.append(x)
        #first input layer to hiiden layer
        arr1 = []
        arr2 = []
        sum_value = 0
        for j in range(self.hidden_units):
            sum_value = 0
            for i in range(len(x)):
               #w*x 
                sum_value += x[i] * self.layers_level[0][0][j][i]
            sum_value += self.layers_level[0][1][j]

            predict = function(self.choose,sum_value)
            arr1.append(sum_value)
            arr2.append(predict)
        self.forward_sum_value.append(arr1)
        self.forward_pre_value.append(arr2)

        #hidden layers
        for i in range(1,self.hidden_layers):
            arr1 = []
            arr2 = []
            for j in range(self.hidden_units):#for rach unit
                sum_value = 0
                for k in range(len(self.layers_level[i][0][j])):
                    # weight * prevalue
                    sum_value += self.layers_level[i][0][j][k] * self.forward_pre_value[i][k]
                    #print('x:',self.layers_level[i][0][j][k],'weight:',self.forward_pre_value[i][k])
                #print('bias:',self.layers_level[i][1][j])
                #plus bias
                sum_value += self.layers_level[i][1][j]
                arr1.append(sum_value)
                #get predict value fron function
                predict = function(self.choose,sum_value)
                arr2.append(predict)
            self.forward_sum_value.append(arr1)
            self.forward_pre_value.append(arr2)

        #outputlayer: 
        arr1 = []
        arr2 = []
        for i in range(self.output_units):
            sum_value = 0
            for j in range(self.hidden_units):
                #print('--L:',sum_value ,self.forward_pre_value[-1][j],self.layers_level[-1][0][i][j])
                sum_value += self.forward_pre_value[-1][j] * self.layers_level[-1][0][i][j]
            sum_value += self.layers_level[-1][1][i]
            #print('bias:',self.layers_level[-1][1][i])
            arr1.append(sum_value)
            #get predict value fron function
            predict = function(self.choose,sum_value)
            arr2.append(predict)
        self.forward_sum_value.append(arr1)
        self.forward_pre_value.append(arr2)
        return self.forward_pre_value[-1]
    
    def update_weight(self,data, ally_true,learn_rate,epochs):

        # weight  self.layers_level[i][0]   bias self.layers_level[i][1]
        for epoch in range(epochs):
            for x, y_true in zip(data, ally_true):
                predicty = self.classify(x)
                '''
                print('---------------weight------------------')
                for i in range(len(self.layers_level)):
                    print(self.layers_level[i][0])
                print('---------------bias------------------')
                for i in range(len(self.layers_level)):
                    print(self.layers_level[i][1])
                print('---------------sum------------------')
                print(self.forward_sum_value)
                print('---------------predict------------------')
                print(self.forward_pre_value)
                '''

                d_L_d_ypreds = []
                for i in range(len(y_true)):
                    d_L_d_ypreds.append(-2 * (y_true[i] - predicty[i]))
                d_L_d_ypred = np.mean(d_L_d_ypreds)
                # last layer
                for j in range(self.output_units):
                    for i in range(len(self.forward_pre_value[-2])):
                        #print(self.forward_pre_value[-2][i],self.forward_pre_value[-1][j])
                        d_ypred_d_w = self.forward_pre_value[-2][i] * deriv_function(self.choose,self.forward_sum_value[-1][j])
                        #weight update
                        self.layers_level[-1][0][j][i] -= learn_rate * d_L_d_ypred * d_ypred_d_w
                    d_ypred_d_b = deriv_function(self.choose,self.forward_sum_value[-1][j])
                    # bias update
                    self.layers_level[-1][1][j] -= learn_rate * d_L_d_ypred * d_ypred_d_b

                # rest layer
                for k in range(len(self.layers_level)-1,0, -1):
                    for j in range(self.hidden_units):
                        for i in range(len(self.forward_pre_value[k-1])):
                            #print('sss',self.forward_pre_value[k-1][i],self.forward_sum_value[k][j])
                            d_ypred_d_w = self.forward_pre_value[k-1][i] * deriv_function(self.choose,self.forward_sum_value[k][j])
                            #print('checkweight',self.layers_level[k-1][0][j][i])
                            self.layers_level[k-1][0][j][i] -= learn_rate * d_L_d_ypred * d_ypred_d_w
                        d_ypred_d_b = deriv_function(self.choose,self.forward_sum_value[k][j])
                        #print('check',self.layers_level[k-1][1][j])
                        #update bias
                        self.layers_level[k-1][1][j] -= learn_rate * d_L_d_ypred * d_ypred_d_b

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.classify, 1, data)
                loss = L2_loss(y_true, y_preds)
                print("Epoch %d L2 loss rate: %.3f" % (epoch, loss))


if __name__ == "__main__":
    main()