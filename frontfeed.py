import board as bd



def main():
    board = [
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]]
    print(bd.minmax(board,1))


if __name__ == "__main__":
    main()