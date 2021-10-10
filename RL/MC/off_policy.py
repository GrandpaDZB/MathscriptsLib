import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from numpy.core.defchararray import array
from numpy.random.mtrand import choice


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


    
def is_valid(state):
    sum_n = np.sum(state)
    if sum_n in [0,1,-1]:
        return True
    else:
        return False


# build states
''' basic_lines = []
for i in [-1,0,1]:
    for j in [-1,0,1]:
        for k in [-1,0,1]:
            basic_lines.append([i,j,k])
line1 = np.copy(basic_lines)
line2 = np.copy(basic_lines)
line3 = np.copy(basic_lines)
states = []
for i in line1:
    for j in line2:
        for k in line3:
            state = np.array([i,j,k])
            if not is_final_state(state) and is_valid(state):
                states.append(state)
with open("./states.pkl", "wb") as f:
    pkl.dump(states, f) '''



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


def run(state, player, actions_player):
    global pi_b
    global pi_w
    if player == 1:
        if np.random.rand() < 0.9:
            state_index = find_state(state, 1)
            a = pi_w[state_index]
            actions_player.append(a)
            state[a[0]][a[1]] = 1
            return state
    else:
        if np.random.rand() < 0.9:
            state_index = find_state(state, -1)
            a = pi_b[state_index]
            actions_player.append(a)
            state[a[0]][a[1]] = -1
            return state
    (x, y) = np.where(state == 0)
    n = len(x)
    choice = np.random.choice(range(n))
    [a, b] = [x[choice], y[choice]]
    actions_player.append([a,b])
    state[a,b] = player
    return state


def zero_number(state):
    n = 0
    for i in range(9):
        if state[i//3, i%3] == 0:
            n += 1
    return n

# default 1 go first
def build_episode():
    board = np.zeros((3,3), dtype='int')
    actions_black = []
    actions_white = []
    states_black = []
    states_white = []
    reward_black = []
    reward_white = []
    player = 1
    while True:
        if player == 1:
            states_white.append(np.copy(board))
            board = run(board, 1, actions_white)
            state_type = is_final_state(board)
            if state_type == 1:
                reward_white.append(1)
                break
            elif state_type == 0:
                reward_white.append(0)
                break
            elif state_type == 2:
                reward_white.append(0)
        else:
            states_black.append(np.copy(board))
            board = run(board, -1, actions_black)
            state_type = is_final_state(board)
            if state_type == -1:
                reward_black.append(1)
                break
            elif state_type == 0:
                reward_black.append(0)
                break
            elif state_type == 2:
                reward_black.append(0)
        player = -player
    return [
        states_black,
        actions_black,
        reward_black,

        states_white,
        actions_white,
        reward_white
        ]
    
# length is 6897
# 黑棋状态空间大小为 2097
# 白棋状态空间大小为 2423
# with open("./states.pkl", "rb") as f:
#     states = pkl.load(f)
with open("./states_black.pkl", "rb") as f:
    states_black = pkl.load(f)
with open("./states_white.pkl", "rb") as f:
    states_white = pkl.load(f)

# init pi

# pi_w = []
# for i in range(2423):
#     ordi = np.where(states_white[i] == 0)
#     pi_w.append([ordi[0][0], ordi[1][0]])
# pi_b = []
# for i in range(2097):
#     ordi = np.where(states_black[i] == 0)
#     pi_b.append([ordi[0][0], ordi[1][0]])

# Q_w = np.zeros((2423, 9))
# C_w = np.zeros((2423, 9))
# for i in range(2423):
#     for j in range(9):
#         if states_white[i][j//3, j%3] != 0:
#             Q_w[i, :][j] = -50
# Q_b = np.zeros((2097, 9))
# C_b = np.zeros((2097, 9))
# for i in range(2097):
#     for j in range(9):
#         if states_black[i][j//3, j%3] != 0:
#             Q_b[i, :][j] = -50
with open("./pi.pkl", "rb") as f:
    [pi_w, pi_b] = pkl.load(f)

with open("./QandC.pkl", "rb") as f:
    [Q_w, C_w, Q_b, C_b] = pkl.load(f)

gamma = 0.8
Q_old_w = np.zeros((2423, 9))
Q_old_b = np.zeros((2097, 9))

max_iteration = 10000
for iteration in range(max_iteration):
    [s_b, a_b, r_b, s_w, a_w, r_w] = build_episode()
    Q_old_w = np.copy(Q_w)
    Q_old_b = np.copy(Q_b)
    pi_w_old = np.copy(pi_w)
    pi_b_old = np.copy(pi_b)
    
    # white player
    G = 0
    T = len(r_w)
    for t in range(T-1, -1, -1):
        state_index = find_state(s_w[t], 1)
        G = gamma*G + r_w[t]
        m = 3*a_w[t][0]+a_w[t][1]
        c = C_w[state_index, m]
        Q_w[state_index, m] = (c/(c+1))*Q_w[state_index, m]+(1/(c+1))*G
        C_w[state_index, m] += 1
        best_action = np.argmax(Q_w[state_index, :])
        pi_w[state_index] = [best_action//3, best_action%3]

    # black player    
    G = 0
    T = len(r_b)
    for t in range(T-1, -1, -1):
        state_index = find_state(s_b[t], -1)
        G = gamma*G + r_b[t]
        m = 3*a_b[t][0]+a_b[t][1]
        c = C_b[state_index, m]
        Q_b[state_index, m] = (c/(c+1))*Q_b[state_index, m]+(1/(c+1))*G
        C_b[state_index, m] += 1
        best_action = np.argmax(Q_b[state_index, :])
        pi_b[state_index] = [best_action//3, best_action%3]

    diff = np.sum(np.abs(Q_old_b - Q_b)) + np.sum(np.abs(Q_old_w - Q_w))
    diff_s = np.sum(np.abs(pi_w_old - pi_w)) + np.sum(np.abs(pi_b_old - pi_b))
    print(f'finish: {(iteration+1)*100/max_iteration}%')
    print(f'diff = {diff}')
    print(f'strategy_change = {diff_s}')


with open("./pi.pkl", "wb") as f:
    pkl.dump([pi_w, pi_b], f)
    print("Save strategy data")

with open("./QandC.pkl", "wb") as f:
    pkl.dump([Q_w, C_w, Q_b, C_b], f)
    print("Save Value Matrix")

# 空棋盘位置在 states_white 1211

