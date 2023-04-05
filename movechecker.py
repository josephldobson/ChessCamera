from movetracker import *


board = chess.Board()
moves = list(board.legal_moves)

lst = []


for i in moves:
    val = 0
    a = board.piece_map()
    a = [a[j] if j in a else 'a' for j in range(64)]

    print(a)
    board.push(i)
    b = board.piece_map()
    print(b)
    b = [b[j] if j in b else 'a' for j in range(64)]
    change = [0 if a[j] == b[j] else 1 for j in range(64)]
    print(change)
    board.pop()
    print('done')