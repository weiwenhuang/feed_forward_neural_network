# Author: Weiwen Huang, Yaxi Huang
# Course: CISC481

import numpy as np
#trainning board for chess game
boards = np.array([
        [[-1,-1,-1],
        [0,0,0],
        [1,1,1]],
        [[-1,-1,-1],
        [1,0,0],
        [0,1,1]],
        [[0,-1,-1],
        [-1,0,0],
        [1,1,1]],
        [[0,-1,-1],
        [1,0,0],
        [1,0,1]],
        [[-1,-1,0],
        [0,0,-1],
        [1,1,1]],
        [[-1,-1,0],
        [0,0,1],
        [1,0,1]],
        [[-1,0,-1],
        [0,0,-1],
        [0,0,1]],
        [[-1,-1,-1],
        [0,1,0],
        [1,0,1]],
        [[-1,-1,-1],
        [0,0,0],
        [1,1,1]],
        [[-1,0,-1],
        [0,-1,0],
        [1,1,1]],
        [[-1,0,-1],
        [0,1,0],
        [1,1,0]],
        [[0,0,-1],
        [1,-1,0],
        [1,0,1]],
        [[0,-1,-1],
        [0,1,0],
        [1,0,0]],
        ])