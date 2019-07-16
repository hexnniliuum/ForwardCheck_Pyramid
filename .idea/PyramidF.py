import time
import random
import numpy as np
import copy

def cFinalTest(square, con, rev):
    for i in range(len(square)):
        big = 0
        counter = 0
        if con[i] != 0 and np.count_nonzero(square[i]) == len(square[i]):
            for j in range(len(square)):
                if rev == 0:
                    if big < square[i][j]:
                        big = square[i][j]
                        counter += 1
                else:
                    if big < square[i][len(square) - j - 1]:
                        big = square[i][len(square) - j - 1]
                        counter += 1
            if counter != con[i]:
                return 0
    return 1

def finalTest(square, col_con, row_con):
    if cFinalTest(square, row_con[0], 0) and \
            cFinalTest(square, row_con[1], 1) and \
            cFinalTest(np.rot90(square), col_con[0], 0) and \
            cFinalTest(np.rot90(square), col_con[1], 1):
        return 1
    return 0


def repeatTest(square):
    for j in range(len(square)):
        check = square[j]
        visited = []
        for i in range(len(check)):
            if check[i] in visited:
                return 0
            if check[i] != 0:
                visited.append(check[i])
        check = square[:, j]
        visited = []
        for i in range(len(check)):
            if check[i] in visited:
                return 0
            if check[i] != 0:
                visited.append(check[i])
    return 1


def firstTest(con):
    for i in range(len(con[0])):
        if con[0][i] == 1 and con[1][i] == 1:
            return 0
        if con[0][i] == len(con[0]):
            if con[1][i] > 1:
                return 0
        if con[1][i] == len(con[1]):
            if con[0][i] > 1:
                return 0
    return 1

def reduceD(square, dom, x, y, col_con, row_con):
    visited = []
    con_f1 = np.flip(col_con[0])
    if x == 0:
        if con_f1[y] == 1:
            visited.extend([1,2,3])
        for i in range(2, len(square) + 1):
            if con_f1[y] == i:
                for j in range(len(square), len(square) - i + 1, -1):
                    visited.append(j)
    if y == 0:
        if row_con[0][y] == 1:
            visited.extend([1,2,3])
        for i in range(2, len(square) + 1):
            if row_con[0][x] == i:
                for j in range(len(square), len(square) - i + 1, -1):
                    visited.append(j)
    for i in set(visited):
        dom.remove(i)
    return dom


def Forward(square, dom, x, y, col_con, row_con):
    if not square[len(square) - 1][len(square) - 1] == 0:
        return square

    x += 1
    if x == len(square):
        x = 0
        y += 1

    if x == 0 | y == 0:
        red_dom = reduceD(square, copy.copy(dom), x, y, col_con, row_con)
    else:
        red_dom = dom

    for i in red_dom:
        square[x][y] = i
        if repeatTest(square) and finalTest(square, col_con, row_con):
            result = Forward(square, dom, x, y, col_con, row_con)
            if result is not None:
                return result
        else:
            square[x][y] = 0


'''print result'''

def joinAndPrint(square, col_con, row_con):
    board = np.zeros(shape=(len(square) + 2, len(square) + 2), dtype=int)
    '''flip column const'''
    con_f1 = np.flip(col_con[1])
    con_f2 = np.flip(col_con[0])

    '''adding constraints'''
    for i in range(len(col_con[0])):
        board[len(square) + 1][i + 1] = con_f1[i]
        board[0][i + 1] = con_f2[i]
        board[i + 1][0] = row_con[0][i]
        board[i + 1][len(square) + 1] = row_con[1][i]
    for i in range(len(square)):
        for j in range(len(square)):
            board[i + 1][j + 1] = square[i][j]

    '''print'''
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] != 0:
                print(board[i][j], end='\t')
            else:
                print("\t", end='')
        print("\n", end='')

    return board



def conFactory(n):
    col = [np.zeros(shape=n, dtype=int), np.zeros(shape=n, dtype=int)]
    row = [np.zeros(shape=n, dtype=int), np.zeros(shape=n, dtype=int)]
    while np.count_nonzero(col[0]) != int(n / 2):
        col[0][random.randint(0, n - 1)] = random.randint(1, n)
    while np.count_nonzero(col[1]) != int(n / 2):
        col[1][random.randint(0, n - 1)] = random.randint(1, n)
    while np.count_nonzero(row[0]) != int(n / 2):
        row[0][random.randint(0, n - 1)] = random.randint(1, n)
    while np.count_nonzero(row[1]) != int(n / 2):
        row[1][random.randint(0, n - 1)] = random.randint(1, n)
    return col, row

def main():
    n = 4
    dom = list(range(1, n + 1))
    col_con = []
    row_con = []
    res = None

    while res is None:
        square = np.zeros(shape=(n, n), dtype=int)
        col_con, row_con = conFactory(n)
        if firstTest(col_con) and firstTest(row_con):
             res = Forward(square, dom, -1, 0, col_con, row_con)

    square = np.zeros(shape=(n, n), dtype=int)
    start = time.time()
    res = Forward(square, dom, -1, 0, col_con, row_con)
    print("time: {:.10f}".format(time.time() - start))
    joinAndPrint(res, col_con, row_con)


if __name__ == '__main__':
    main()
