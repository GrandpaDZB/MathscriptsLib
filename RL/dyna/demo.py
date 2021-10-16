import numpy as np
import matplotlib.pyplot as plt

best_trace = [[2, 0],
 [2, 1],
 [3, 1],
 [4, 1],
 [4, 2],
 [4, 3],
 [4, 4],
 [3, 4],
 [3, 5],
 [3, 6],
 [3, 7],
 [3, 8],
 [2, 8],
 [1, 8],
 [0, 8]]

class world:
    def __init__(self):
        self.state = [2, 0]
        self.blocks = [
            [[x, 2] for x in range(1,4)],
            [[4, 5]],
            [[x, 7] for x in range(0,3)]
        ]
        self.goal = [0, 8]

        self.notvisit = np.zeros((6, 9))

        self.Q = np.zeros((6,9,4))
        self.epsilon = 0.3
        self.k = 0.05
        pass

    def is_valid(self, state, motion):
        new_state = [state[0], state[1]]
        if motion == 0:
            new_state[1] += 1
        elif motion == 1:
            new_state[1] -= 1
        elif motion == 2:
            new_state[0] += 1
        elif motion == 3:
            new_state[0] -= 1

        if new_state[0] < 0 or new_state[0] > 5 or new_state[1] < 0 or new_state[1] > 8:
            return [False, new_state]
        for each in self.blocks:
            for block in each:
                if new_state == block:
                    return [False, new_state]
        return [True, new_state]

    def drawTrace(self, trace):
        trace_map = np.zeros((6, 9))
        for each in self.blocks:
            for block in each:
                trace_map[block[0], block[1]] = -1
        for each in trace:
            trace_map[each[0], each[1]] = 1
        print(trace_map)

    def best_trace(self):
        state = [2, 0]
        trace = [[2, 0]]
        while state != self.goal:
            action = np.argmax(self.Q[state[0], state[1], :])
            [new_state, _] = self.move(state, action)
            state = [new_state[0], new_state[1]]
            trace.append([new_state[0], new_state[1]])
        return trace
        
    def move(self, state, action):
        [is_valid, new_state] = self.is_valid(state, action)
        if not is_valid:
            new_state = [state[0], state[1]]
        if new_state == self.goal:
            reward = 10
        else:
            reward = self.k*np.math.sqrt(self.notvisit[new_state[0],new_state[1]])
        return [new_state, reward]

    def pi(self, state, add_motivation):
        rand_num = np.random.rand()
        if rand_num < 1-self.epsilon:
            if add_motivation:
                tmp_Q = np.copy(self.Q[state[0], state[1], :])
                for i in range(4):
                    [new_state, _] = self.move(state, i)
                    tmp_Q[i] += self.k*np.math.sqrt(self.notvisit[new_state[0],new_state[1]])
                action = np.argmax(tmp_Q)
            else:
                action = np.argmax(self.Q[state[0], state[1], :])
        else:
            action = np.random.choice([0,1,2,3])
        return action


    def build_episode(self):
        state = [2,0]
        time_step = 0
        while state != self.goal:
            action = self.pi(state, False)
            [new_state, reward] = self.move(state, action)
            time_step += 1
            self.Q[state[0], state[1], action] += alpha*(reward + gamma*np.max(self.Q[new_state[0], new_state[1], :]) - self.Q[state[0], state[1], action])
            state = [new_state[0], new_state[1]]
        return time_step

alpha = 0.8
gamma = 0.9
max_iteration = 50

plt.plot(np.ones(max_iteration)*15, 'r--')

evaluation = [[],[],[]]

for times in range(1):
    # direct RL
    model1 = world()
    history1 = []
    for t in range(max_iteration):
        if t == int(max_iteration/2):
            model1.blocks[2] = model1.blocks[2][1:]
        history1.append(model1.build_episode())
    trace = model1.best_trace()
    evaluation[0].append(len(trace))
    print(f'trace_len = {len(trace)}')
    model1.drawTrace(trace)
    plt.plot(history1)

    # dyna 5
    model2 = world()
    history2 = []
    n = 20
    for t in range(max_iteration):
        if t == int(max_iteration/2):
            model2.blocks[2] = model2.blocks[2][1:]
        S_before = []
        A_before = []
        state = [2, 0]
        time_step = 0
        while state != model2.goal:
            if state not in S_before:
                S_before.append(state)
                A_before.append([])
            action = model2.pi(state, False)
            index = S_before.index(state)
            if action not in A_before[index]:
                A_before[index].append(action)
            [new_state, reward] = model2.move(state, action)
            # model2.notvisit += 1
            # model2.notvisit[new_state[0],new_state[1]] = 0
            time_step += 1
            model2.Q[state[0], state[1], action] += alpha*(reward + gamma*np.max(model2.Q[new_state[0], new_state[1], :]) - model2.Q[state[0], state[1], action])
            state = [new_state[0], new_state[1]]
            for _ in range(n):
                index = np.random.randint(len(S_before))
                tmp_state = S_before[index]
                tmp_action = np.random.choice(A_before[index])
                [tmp_new_state, tmp_reward] = model2.move(tmp_state, tmp_action)
                model2.Q[tmp_state[0],tmp_state[1],tmp_action] += alpha*(tmp_reward + gamma*np.max(model2.Q[tmp_new_state[0], tmp_new_state[1], :]) - model2.Q[tmp_state[0], tmp_state[1], tmp_action])
        history2.append(time_step)
    trace = model2.best_trace()
    evaluation[1].append(len(trace))
    print(f'trace_len = {len(trace)}')
    model2.drawTrace(trace)
    plt.plot(history2)

    # dyna 20 with motivation
    model3 = world()
    history3 = []
    n = 20
    for t in range(max_iteration):
        if t == int(max_iteration/2):
            model3.blocks[2] = model3.blocks[2][1:]
        S_before = []
        A_before = []
        state = [2, 0]
        time_step = 0
        while state != model3.goal:
            if state not in S_before:
                S_before.append(state)
                A_before.append([])
            action = model3.pi(state, False)
            index = S_before.index(state)
            if action not in A_before[index]:
                A_before[index].append(action)
            [new_state, reward] = model3.move(state, action)
            model3.notvisit += 1
            model3.notvisit[new_state[0],new_state[1]] = 0
            time_step += 1
            model3.Q[state[0], state[1], action] += alpha*(reward + gamma*np.max(model3.Q[new_state[0], new_state[1], :]) - model3.Q[state[0], state[1], action])
            state = [new_state[0], new_state[1]]
            for _ in range(n):
                index = np.random.randint(len(S_before))
                tmp_state = S_before[index]
                tmp_action = np.random.choice(A_before[index])
                [tmp_new_state, tmp_reward] = model3.move(tmp_state, tmp_action)
                model3.Q[tmp_state[0],tmp_state[1],tmp_action] += alpha*(tmp_reward + gamma*np.max(model3.Q[tmp_new_state[0], tmp_new_state[1], :]) - model3.Q[tmp_state[0], tmp_state[1], tmp_action])
        history3.append(time_step)
    trace = model3.best_trace()
    evaluation[2].append(len(trace))
    print(f'trace_len = {len(trace)}')
    model3.drawTrace(trace)
    plt.plot(history3)
            
    print(times)