import cnn
import numpy as np
import data as da
import board as bd

def main():

    #cnn.addtest()
    #cnn.defhexapawntest()
    cnn.testweight()

def defhexapawntest():
    a = cnn.Neuron('sigmoid',10,2,2,1)
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

def addtest():

    a = cnn.Neuron('sigmoid',2,2,2,2)


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


def testweight():
    a = cnn.Neuron('sigmoid',2,2,2,1)
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

    a.update_weight(data,all_y_trues)
    print(a.feedforward(np.array([25,6])))


if __name__ == "__main__":
    main()