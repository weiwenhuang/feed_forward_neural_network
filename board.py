import copy
# part 1 

#return movable chess pieces
#return type: arraylist
def TOMOVE(state,player):
    res =  []
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == player:
                res.append([i,j])
    return res

#return all posibble move for player
#return type: hashmap
def ACTIONS(state,player):
    queue = TOMOVE(state,player)
    res = {}
    while queue:
        index = queue.pop(0)
        strindex = tostr(index)
        if player == 1:
            if index[0]-1 in range(0,len(state)+1): # can move
                if state[index[0]-1][index[1]] == 0: # move on step
                    if strindex not in res:
                        res[strindex] = [[index[0]-1,index[1]]]
                    else:
                        res[strindex].append([index[0]-1,index[1]])
                if index[1]-1 in range(0,len(state[0])):
                    if state[index[0]-1][index[1]-1] == player*-1:
                        if strindex not in res:
                            res[strindex] = [[index[0]-1,index[1]-1]]
                        else:
                            res[strindex].append([index[0]-1,index[1]-1])
                if index[1]+1 in range(0,len(state[0])):
                    if state[index[0]-1][index[1]+1] == player*-1:
                        if strindex not in res:
                            res[strindex] = [[index[0]-1,index[1]+1]]
                        else:
                            res[strindex].append([index[0]-1,index[1]+1])
        elif player == -1:
            if index[0]+1 in range(0,len(state)+1): # can move
                if state[index[0]+1][index[1]] == 0: # move on step
                    if strindex not in res:
                        res[strindex] = [[index[0]+1,index[1]]]
                    else:
                        res[strindex].append([index[0]+1,index[1]])
                if index[1]-1 in range(0,len(state[0])):
                    if state[index[0]+1][index[1]-1] == player*-1:
                        if strindex not in res:
                            res[strindex] = [[index[0]+1,index[1]-1]]
                        else:
                            res[strindex].append([index[0]+1,index[1]-1])
                if index[1]+1 in range(0,len(state[0])):
                    if state[index[0]+1][index[1]+1] == player*-1:
                        if strindex not in res:
                            res[strindex] = [[index[0]+1,index[1]+1]]
                        else:
                            res[strindex].append([index[0]+1,index[1]+1])
    return res

#implement the action
#return type array list
def RESULT(origion,act):
    state = copy.deepcopy(origion)
    s = state[act[0][0]][act[0][1]] 
    state[act[0][0]][act[0][1]] = 0
    state[act[1][0]][act[1][1]] = s
    return state

#check is thhe state terminal
#return type Boolean
def IS_TERMINAL(state,player):
    for i in state[0]:
        if i == 1:
            return True, 1
    for i in state[2]:
        if i == -1:
            return True, -1
    if not ACTIONS(state,player):
        return True, player*-1
    return False,0

def utility(winner):
    if winner == 1:
        return 1
    else: 
        return -1 

#change array to str
def tostr(arr):
    if len(arr) == 2:
        return str(arr[0]) +','+ str(arr[1])
    else:
        return print("arr lenth must = 2")

#change str to arr
def toarr(str):
    arr = str.split(',')
    return [int(arr[0]),int(arr[1])]

#part2
#minmax algotithm
# return value
def minmax(state,player):
    r,w = IS_TERMINAL(state,player)
    if r:
        return utility(w)
    if player == 1:
        value = -1000
        action = ACTIONS(state,player)
        for i in action:
            act = [toarr(i),action[i][0]]
            value = max(value,minmax(RESULT(state,act),player*-1))
        return value
    if player == -1:
        value = 1000
        action = ACTIONS(state,player)
        for i in action:
            act = [toarr(i),action[i][0]]
            value = min(value,minmax(RESULT(state,act),player*-1))
        return value
    
def main():
    board = [
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]]
    print(minmax(board,1))

if __name__ == "__main__":
    main()