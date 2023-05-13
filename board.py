
def TOMOVE(state):
    res =  []
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 1:
                res.append([i,j])
    return res

def ACTIONS(state):
    queue = TOMOVE(state)
    res = {}
    while queue:
        index = queue.pop(0)
        strindex = tostr(index)
        if index[0]-1 in range(0,len(state)+1): # can move
            if state[index[0]-1][index[1]] == 0: # move on step
                if strindex not in res:
                    res[strindex] = [[index[0]-1,index[1]]]
                else:
                    res[strindex].append([index[0]-1,index[1]])
            if index[1]-1 in range(0,len(state[0])):
                if state[index[0]-1][index[1]-1] == -1:
                    if strindex not in res:
                        res[strindex] = [[index[0]-1,index[1]-1]]
                    else:
                        res[strindex].append([index[0]-1,index[1]-1])
            if index[1]+1 in range(0,len(state[0])):
                if state[index[0]-1][index[1]+1] == -1:
                    if strindex not in res:
                        res[strindex] = [[index[0]-1,index[1]+1]]
                    else:
                        res[strindex].append([index[0]-1,index[1]+1])
    return res


def tostr(arr):
    if len(arr) == 2:
        return str(arr[0]) +','+ str(arr[1])
    else:
        return print("arr lenth must = 2")
    

def main():
    board = [[-1,-1,-1],[0,-1,0],[1,1,1]]
    print(ACTIONS(board))


if __name__ == "__main__":
    main()