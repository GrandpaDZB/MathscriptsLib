import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl



class model:
    def __init__(self):
        # 0-A
        # 1-B
        self.state = 0
        self.Q = [0,0]
        self.Q2 = [0,0]
        pass

    def get_action(self, epsilon):
        rand_num = np.random.rand()
        if rand_num < 1-epsilon:
            if self.Q[0] >= self.Q[1]:
                return 0
            else:
                return 1
        else:
            if self.Q[0] < self.Q[1]:
                return 0
            else:
                return 1
    
    def get_action_d(self, epsilon):
        rand_num = np.random.rand()
        if rand_num < 1-epsilon:
            if self.Q[0]+self.Q2[0] >= self.Q[1]+self.Q2[1]:
                return 0
            else:
                return 1
        else:
            if self.Q[0]+self.Q2[0] < self.Q[1]+self.Q2[1]:
                return 0
            else:
                return 1

    # 0-left
    # 1-right
    def move(self, action):
        if action == 0:
            return np.random.normal(-0.1, 1.0)
        else:
            return 0

max_iteration = 10000
learning_rate = 0.002
epsilon = 0.1

# Q-learning
model_QL = model()
action = model_QL.get_action(epsilon)
history = []
for i in range(max_iteration):
    reward = model_QL.move(action)
    new_action = model_QL.get_action(epsilon)
    model_QL.Q[action] += learning_rate*(reward + 0 - model_QL.Q[action])
    action = new_action
    history.append(model_QL.Q[0] - model_QL.Q[1])
plt.plot(history)


# double Q-learning
model_dQ = model()
action = model_QL.get_action_d(epsilon)
history = []
for i in range(max_iteration):
    reward = model_QL.move(action)
    new_action = model_QL.get_action(epsilon)
    if np.random.rand() < 0.5:
        model_QL.Q[action] += learning_rate*(reward + 0 - model_QL.Q[action])
    else:
        pass
    action = new_action
    history.append(model_QL.Q[0] - model_QL.Q[1])
plt.plot(history)
