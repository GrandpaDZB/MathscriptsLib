import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from numpy.core.defchararray import array


def is_final_state(state):
    for i in range(3):
        line = state[:, i]
        if line[0] != 0 and line[0] == line[1] and line[0] == line[2]:
            return True
    for i in range(3):
        line = state[i, :]
        if line[0] != 0 and line[0] == line[1] and line[0] == line[2]:
            return True
    if state[0, 0] != 0 and state[0, 0] == state[1, 1] and state[0, 0] == state[2, 2]:
        return True
    if not np.any(state == 0):
        return True
    return False

def is_win_state(state):
    for i in range(3):
        line = state[:, i]
        if line[0] != 0 and line[0] == line[1] and line[0] == line[2]:
            return True
    for i in range(3):
        line = state[i, :]
        if line[0] != 0 and line[0] == line[1] and line[0] == line[2]:
            return True
    if state[0, 0] != 0 and state[0, 0] == state[1, 1] and state[0, 0] == state[2, 2]:
        return True
    return False
    
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

# length is 6897
with open("./states.pkl", "rb") as f:
    states = pkl.load(f)

# init pi
pi = []
for i in range(6897):
    ordi = np.where(states[i] == 0)
    pi.append([ordi[0][0], ordi[1][0]])
pi_f = np.copy(pi)
pi_s = np.copy(pi)


# motion strategy
def strategy_b(state):
    (x, y) = np.where(state == 0)
    n = len(x)
    choice = np.random.choice(range(n))
    return [x[choice], y[choice]]

def run(state, player, actions_player):
    [x,y] = strategy_b(state)
    actions_player.append([x,y])
    state[x,y] = player
    return state

def find_state(state):
    global states
    for i in range(6897):
        if np.all(states[i] == state):
            return i
    return -1

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
    while not is_final_state(board):
        if player == 1:
            states_white.append(np.copy(board))
            board = run(board, 1, actions_white)
            reward_white.append(0)
        else:
            states_black.append(np.copy(board))
            board = run(board, -1, actions_black)
            reward_black.append(0)
        player = -player
    if is_win_state(board):
        if -player == 1:
            reward_white[-1] = 1
        else:
            reward_black[-1] = 1
    return [
        states_black,
        actions_black,
        reward_black,

        states_white,
        actions_white,
        reward_white
        ]
    
        
with open("./pi.pkl", "rb") as f:
    [pi_white, pi_black] = pkl.load(f)
# pi_black = [0 for x in range(6897)]
# pi_white = [0 for x in range(6897)]
with open("./QandC.pkl", "rb") as f:
    [Q, C] = pkl.load(f)
# Q = np.zeros((6897, 9))
# C = np.zeros((6897, 9))

Q_old = np.zeros((6897, 9))

max_iteration = 10000
for iteration in range(max_iteration):
    [s_b, a_b, r_b, s_w, a_w, r_w] = build_episode()
    Q_old = np.copy(Q)
    # white player
    G, W = 0, 1
    T = len(r_w)
    for t in range(T-1, -1, -1):
        state_index = find_state(s_w[t])
        G += r_w[t]
        C[state_index, 3*a_w[t][0]+a_w[t][1]] += W
        Q[state_index, 3*a_w[t][0]+a_w[t][1]] += (W/C[state_index, 3*a_w[t][0]+a_w[t][1]])*(G-Q[state_index, 3*a_w[t][0]+a_w[t][1]])
        best_action = np.argmax(Q[state_index, :])
        pi_white[state_index] = best_action
        if best_action != 3*a_w[t][0]+a_w[t][1]:
            break
        W = W*(1/(zero_number(s_w[t])))
    # black player    
    G, W = 0, 1
    T = len(r_b)
    for t in range(T-1, -1, -1):
        state_index = find_state(s_b[t])
        G += r_b[t]
        C[state_index, 3*a_b[t][0]+a_b[t][1]] += W
        Q[state_index, 3*a_b[t][0]+a_b[t][1]] += (W/C[state_index, 3*a_b[t][0]+a_b[t][1]])*(G-Q[state_index, 3*a_b[t][0]+a_b[t][1]])
        best_action = np.argmax(Q[state_index, :])
        pi_black[state_index] = best_action
        if best_action != 3*a_b[t][0]+a_b[t][1]:
            break
        W = W*(1/(zero_number(s_b[t])))
    diff = np.sum(np.abs(Q_old - Q))
    print(f'finish: {(iteration+1)*100/max_iteration}%')
    print(f'diff = {diff}')
with open("./pi.pkl", "wb") as f:
    pkl.dump([pi_white, pi_black], f)
    print("Save strategy data")
with open("./QandC.pkl", "wb") as f:
    pkl.dump([Q, C], f)
    print("Save Value matrix")
# 空棋盘位置在3448

