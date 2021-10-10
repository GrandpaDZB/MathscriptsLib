import numpy as np
import pickle as pkl

# 判断状态是否为结束状态，运行效率比遍历集合要快，可以把返回值修改为int，表明状态类型
# 返回1表示  白棋获胜
# 返回-1表示 黑棋获胜
# 返回0表示  平局
# 返回2表示  游戏在进行
def is_final_state(state):
    for i in range(3):
        line = state[:, i]
        if line[0] != 0 and line[0] == line[1] and line[0] == line[2]:
            if line[0] == 1:
                return 1
            else:
                return -1
    for i in range(3):
        line = state[i, :]
        if line[0] != 0 and line[0] == line[1] and line[0] == line[2]:
            if line[0] == 1:
                return 1
            else:
                return -1
    if state[0, 0] != 0 and state[0, 0] == state[1, 1] and state[0, 0] == state[2, 2]:
        if state[0, 0] == 1:
            return 1
        else:
            return -1
    if state[0, 2] != 0 and state[0, 2] == state[1, 1] and state[0, 2] == state[2, 0]:
        if state[0, 2] == 1:
            return 1
        else:
            return -1
    if not np.any(state == 0):
        return 0
    return 2
def zero_number(state):
    n = 0
    for i in range(9):
        if state[i//3, i%3] == 0:
            n += 1
    return n


def find_state(state, player):
    global states_white
    global states_black
    if player == 1:
        for i in range(2423):
            if np.all(states_white[i] == state):
                return i
    else:
        for i in range(2097):
            if np.all(states_black[i] == state):
                return i
    return -1

with open("./states_black.pkl", "rb") as f:
    states_black = pkl.load(f)
with open("./states_white.pkl", "rb") as f:
    states_white = pkl.load(f)
with open("./pi.pkl", "rb") as f:
    [pi_w, pi_b] = pkl.load(f)
with open("./QandC.pkl", "rb") as f:
    [Q_w, C_w, Q_b, C_b] = pkl.load(f)


board = np.zeros((3,3))

is_player_white = int(input("is_player_white = "))




if is_player_white == 1:
    while True:
        print("========================================")
        position = input("Position: ")
        x = int(position[1])-1
        y = int(position[3])-1
        board[x, y] = 1
        state_type = is_final_state(board)
        print(board)

        if state_type == 1:
            print("=============================")
            print("You Win")
            break
        elif state_type == 0:
            print("=============================")
            print("Tie")
            break

        state_index = find_state(board, -1)
        AI_position = pi_b[state_index]
        board[AI_position[0], AI_position[1]] = -1
        state_type = is_final_state(board)
        print(board)

        if state_type == -1:
            print("=============================")
            print("You Lose")
            break
        elif state_type == 0:
            print("=============================")
            print("Tie")
            break
else:
    while True:
        print("========================================")
        state_index = find_state(board, 1)
        AI_position = pi_w[state_index]
        board[AI_position[0], AI_position[1]] = 1
        state_type = is_final_state(board)
        print(board)

        if state_type == 1:
            print("=============================")
            print("You Lose")
            break
        elif state_type == 0:
            print("=============================")
            print("Tie")
            break

        position = input("Position: ")
        x = int(position[1])-1
        y = int(position[3])-1
        board[x, y] = -1
        state_type = is_final_state(board)
        print(board)

        if state_type == -1:
            print("=============================")
            print("You Win")
            break
        elif state_type == 0:
            print("=============================")
            print("Tie")
            break
