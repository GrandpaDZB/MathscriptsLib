import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand
import pickle as pkl

class model:
    def __init__(self):
        self.cols = 10
        self.rows = 7
        self.position = np.array([int(self.rows/2),int(self.cols/4)], dtype="int")
        self.Q = np.zeros((self.rows,self.cols,9))
        with open("./Q.pkl", "rb") as f:
            self.Q = pkl.load(f)
        self.epsilon = 0.1
        self.learning_rate = 0.3
        self.gamma = 1.0
        self.history = []
        self.preprocess_Q()
        pass

    def preprocess_Q(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if i == 0:
                    self.Q[i,j,0] = -10000
                    self.Q[i,j,1] = -10000
                    self.Q[i,j,2] = -10000
                if i == self.rows-1:
                    self.Q[i,j,6] = -10000
                    self.Q[i,j,7] = -10000
                    self.Q[i,j,8] = -10000
                if j == 0:
                    self.Q[i,j,0] = -10000
                    self.Q[i,j,3] = -10000
                    self.Q[i,j,6] = -10000
                if j == self.cols-1:
                    self.Q[i,j,2] = -10000
                    self.Q[i,j,5] = -10000
                    self.Q[i,j,8] = -10000                   

    def init_position(self):
        self.position = np.array([int(self.rows/2),int(self.cols/4)], dtype="int")
        return 

    def get_action(self, epsilon):
        rand_num = np.random.rand()
        if rand_num < 1-epsilon:
            action = np.argmax(self.Q[self.position[0],self.position[1],:])
        else:
            k = [0,1,2,3,4,5,6,7,8]
            if self.position[0] == 0:
                k = [x for x in k if x not in [0,1,2]]
            if self.position[0] == self.rows-1:
                k = [x for x in k if x not in [6,7,8]]
            if self.position[1] == 0:
                k = [x for x in k if x not in [0,3,6]]
            if self.position[1] == self.cols-1:
                k = [x for x in k if x not in [2,5,8]]
            action = np.random.choice(k)
        return int(action)

    def move(self, action):
        new_position = self.position + np.array([action//3-1, action%3-1])
        # add wind interference
        if new_position[1] >= int(self.cols/2):
            rand_num = np.random.rand()
            if rand_num < 0.333:
                pass
            elif rand_num >= 0.333 and rand_num < 0.666:
                new_position[0] -= 1
            else:
                new_position[0] -= 2
        if new_position[0] < 0:
            new_position[0] = 0
        # count reward
        if new_position[0] == int(self.rows/2) and new_position[1] == int(self.cols*0.75):
            reward = 0
        elif new_position[0] < 0 or new_position[0] > self.rows-1 or new_position[1] < 0 or new_position[1] > self.cols-1:
            reward = -100
        else:
            reward = -1
        self.position = new_position
        return reward
    
    def build_episode(self):
        steps = 0
        is_done = 0
        self.init_position()
        reward = -1
        action = self.get_action(self.epsilon)
        while(reward == -1):
            old_state = np.copy(self.position)
            reward = self.move(action)
            steps += 1
            new_Q = 0
            if reward == -1:
                new_action = self.get_action(self.epsilon)
                new_Q = self.Q[self.position[0],self.position[1],int(new_action)]
            elif reward == -100:
                new_Q = -10000
            else:
                new_Q = 0
                new_action = np.random.choice([0,1,2,3,4,5,6,7,8])
            self.Q[old_state[0], old_state[1], action] += self.learning_rate*(reward + self.gamma*new_Q-self.Q[old_state[0], old_state[1], action])
            action = new_action
        if reward == 0:
            is_done = 1
        print(f'steps: {steps}\tis_done: {is_done}')
        self.history.append(steps)

    def display(self, need_plot):
        trace = []
        self.init_position()
        trace.append(np.copy(self.position))
        reward = -1
        action = self.get_action(self.epsilon)
        while(reward == -1):
            old_state = np.copy(self.position)
            reward = self.move(action)
            trace.append(np.copy(self.position))
            new_Q = 0
            if reward == -1:
                new_action = self.get_action(self.epsilon)
                new_Q = self.Q[self.position[0],self.position[1],int(new_action)]
            elif reward == -100:
                new_Q = -10000
            else:
                new_Q = 0
                new_action = np.random.choice([0,1,2,3,4,5,6,7,8])
            self.Q[old_state[0], old_state[1], action] += self.learning_rate*(reward + self.gamma*new_Q-self.Q[old_state[0], old_state[1], action])
            action = new_action
        if need_plot:
            print(trace)
            x = []
            y = []
            for each in trace:
                x.append(each[0])
                y.append(each[1])
            plt.plot(x, y, "-*")
        return len(trace)

world = model()


max_iteration = 100000
for i in range(max_iteration):
    old_Q = np.copy(world.Q)
    world.build_episode()
    Q = world.Q
    diff = np.abs(Q-old_Q)
    chg = np.sum(diff)/np.sum(np.abs(Q))
    print(f'diff: {chg*100}%')
    print(f'finish: {(i+1)*100/max_iteration}%')
with open("./Q.pkl", "wb") as f:
    pkl.dump(world.Q, f)

# evaluate
length_mean = 0
length_min = 100
for i in range(1000):
    length = world.display(False)
    length_mean = (i/(i+1))*length_mean + (1/(i+1))*length
    if length < length_min:
        length_min = length
print(f'Mean step: {length_mean}')
print(f'Min step: {length_min}')



