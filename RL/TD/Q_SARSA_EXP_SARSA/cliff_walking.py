import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl



class world:
    def __init__(self):
        self.position = np.array([3,0], dtype='int')
        self.last_position = np.copy(self.position)
        self.Q = np.zeros((4,6,4))
        # with open("./Q_sarsa.pkl", "rb") as f:
        #     self.Q = pkl.load(f)
        # with open("./Q_QL.pkl", "rb") as f:
        #     self.Q = pkl.load(f)
        self.is_done = False
        # self.init_Q()

    def init_position(self):
        self.position = np.array([3,0], dtype='int')
        self.is_done = False

    def init_Q(self):
        for i in range(4):
            for j in range(6):
                if i == 0:
                    self.Q[i,j,0] = -1000
                if (i == 2 and (j != 0 and j != 5)) or (i == 3 and (j == 0 or j == 5)):
                    self.Q[i,j,2] = -1000
                if j == 0:
                    self.Q[i,j,3] = -1000
                if j == 5:
                    self.Q[i,j,1] = -1000
                

    def get_action(self, epsilon):
        rand_num = np.random.rand()
        if rand_num < 1 - epsilon:
            action = np.argmax(self.Q[self.position[0],self.position[1],:])
        else:
            action = np.random.choice([0,1,2,3])
        return int(action)

    def move(self, action):
        self.last_position = np.copy(self.position)
        if action == 0:
            self.position[0] -= 1
        elif action == 1:
            self.position[1] += 1
        elif action == 2:
            self.position[0] += 1
        else:
            self.position[1] -= 1
        if self.position[0] == 3 and self.position[1] == 5:
            self.is_done = True
            return 0
        elif self.position[0] < 0 or self.position[0] > 3 or self.position[1] < 0 or self.position[1] > 5:
            self.init_position()
            return -100
        elif self.position[0] == 3 and self.position[1] in [1,2,3,4]:
            self.init_position()
            return -100
        else:
            return -1

class SARSA:
    def __init__(self):
        self.learning_rate = 0.6
        pass

    def run(self, s1, a1, r2, s2, a2, Q, is_final):
        if not is_final:
            Q[s1[0], s1[1], a1] += self.learning_rate*(r2 + Q[s2[0], s2[1], a2] - Q[s1[0], s1[1], a1])
        else:
            Q[s1[0], s1[1], a1] += self.learning_rate*(r2 + 0 - Q[s1[0], s1[1], a1])
        # else:
        #     Q[s1[0], s1[1], a1] += self.learning_rate*(r2 - 100 - Q[s1[0], s1[1], a1])
        return

class QL:
    def __init__(self):
        self.learning_rate = 0.6
        pass

    def run(self, s1, a1, r2, s2, a2, Q, is_final):
        if not is_final:
            Q[s1[0], s1[1], a1] += self.learning_rate*(r2 + np.max(Q[s2[0], s2[1], :]) - Q[s1[0], s1[1], a1])
        else:
            Q[s1[0], s1[1], a1] += self.learning_rate*(r2 + 0 - Q[s1[0], s1[1], a1])
        # else:
        #     Q[s1[0], s1[1], a1] += self.learning_rate*(r2 - 100 - Q[s1[0], s1[1], a1])
        return

class exp_SARSA:
    def __init__(self):
        self.learning_rate = 0.6
        pass

    def run(self, s1, a1, r2, s2, a2, Q, is_final):
        global epsilon
        expectation = (1-epsilon)*Q[s2[0], s2[1],np.argmax(Q[s2[0], s2[1], :])] + epsilon*(0.25*np.sum(Q[s2[0],s2[1],:]))
        if not is_final:
            Q[s1[0], s1[1], a1] += self.learning_rate*(r2 + expectation - Q[s1[0], s1[1], a1])
        else:
            Q[s1[0], s1[1], a1] += self.learning_rate*(r2 + 0 - Q[s1[0], s1[1], a1])
        # else:
        #     Q[s1[0], s1[1], a1] += self.learning_rate*(r2 - 100 - Q[s1[0], s1[1], a1])
        return


epsilon = 0.1
max_iteration = 1000
need_plot = False

model_sarsa = SARSA()
world_sarsa = world()

model_QL = QL()
world_QL = world()

model_ex = exp_SARSA()
world_ex = world()

convergence_time_step = [0,0,0]
# ===================================== SARSA ==================

for i in range(max_iteration):    
    steps = 0
    trace = []
    world_sarsa.init_position()
    trace.append(np.copy(world_sarsa.position))
    action = world_sarsa.get_action(epsilon)
    old_Q = np.copy(world_sarsa.Q)
    while not world_sarsa.is_done:
        reward = world_sarsa.move(action)
        trace.append(np.copy(world_sarsa.position))
        steps += 1
        if need_plot:
            print(world_sarsa.position)
            print(action)
            print(reward)
        if not world_sarsa.is_done:
            new_action = world_sarsa.get_action(epsilon)
        else:
            new_action = np.random.choice([0,1,2,3])
        model_sarsa.run(world_sarsa.last_position, action, reward, world_sarsa.position, new_action, world_sarsa.Q, world_sarsa.is_done)
        action = new_action
    diff = np.abs(old_Q - world_sarsa.Q)
    diff = np.sum(diff)*100/np.sum(np.abs(old_Q))
    if diff < 5.0:
        convergence_time_step[0] += 1
        if convergence_time_step == 5:
            convergence_time_step = i - 4
    else:
        convergence_time_step[0] = 0
    print(f'steps = {steps}')
    print(f'diff = {diff}%')
    print(f'finish = {(i+1)*100/max_iteration}%')
    if need_plot:
        x = []
        y = []
        for each in trace:
            x.append(each[1])
            y.append(each[0])
        plt.plot(x,y,"-*")
            
with open("./Q_sarsa.pkl", "wb") as f:
    pkl.dump(world_sarsa.Q, f)

# ===================================== QL ==================

for i in range(max_iteration):
    steps = 0
    trace = []
    world_QL.init_position()
    trace.append(np.copy(world_QL.position))
    action = world_QL.get_action(epsilon)
    old_Q = np.copy(world_QL.Q)
    while not world_QL.is_done:
        reward = world_QL.move(action)
        trace.append(np.copy(world_QL.position))
        steps += 1
        if need_plot:
            print(world_QL.position)
            print(action)
            print(reward)
        if not world_QL.is_done:
            new_action = world_QL.get_action(epsilon)
        else:
            new_action = np.random.choice([0,1,2,3])
        model_QL.run(world_QL.last_position, action, reward, world_QL.position, new_action, world_QL.Q, world_QL.is_done)
        action = new_action
    diff = np.abs(old_Q - world_QL.Q)
    diff = np.sum(diff)*100/np.sum(np.abs(old_Q))
    if diff < 5.0:
        convergence_time_step[1] += 1
        if convergence_time_step == 5:
            convergence_time_step = i - 4
    else:
        convergence_time_step[1] = 0
    print(f'steps = {steps}')
    print(f'diff = {diff}%')
    print(f'finish = {(i+1)*100/max_iteration}%')
    if need_plot:
        x = []
        y = []
        for each in trace:
            x.append(each[1])
            y.append(each[0])
        plt.plot(x,y,"-*")
            
with open("./Q_QL.pkl", "wb") as f:
    pkl.dump(world_QL.Q, f)

# ================================ exp SARSA ============

for i in range(max_iteration):
    steps = 0
    trace = []
    world_ex.init_position()
    trace.append(np.copy(world_ex.position))
    action = world_ex.get_action(epsilon)
    old_Q = np.copy(world_ex.Q)
    while not world_ex.is_done:
        reward = world_ex.move(action)
        trace.append(np.copy(world_ex.position))
        steps += 1
        if need_plot:
            print(world_ex.position)
            print(action)
            print(reward)
        if not world_ex.is_done:
            new_action = world_ex.get_action(epsilon)
        else:
            new_action = np.random.choice([0,1,2,3])
        model_ex.run(world_ex.last_position, action, reward, world_ex.position, new_action, world_ex.Q, world_ex.is_done)
        action = new_action
    diff = np.abs(old_Q - world_ex.Q)
    diff = np.sum(diff)*100/np.sum(np.abs(old_Q))
    if diff < 5.0:
        convergence_time_step[2] += 1
        if convergence_time_step == 5:
            convergence_time_step = i - 4
    else:
        convergence_time_step[2] = 0
    print(f'steps = {steps}')
    print(f'diff = {diff}%')
    print(f'finish = {(i+1)*100/max_iteration}%')
    if need_plot:
        x = []
        y = []
        for each in trace:
            x.append(each[1])
            y.append(each[0])
        plt.plot(x,y,"-*")
            
with open("./Q_ex.pkl", "wb") as f:
    pkl.dump(world_ex.Q, f)


# ======================= test =================
min_steps = [100,100,100]
mean_steps = [0,0,0]

for i in range(1000):    
    steps = 0
    world_sarsa.init_position()
    trace.append(np.copy(world_sarsa.position))
    action = world_sarsa.get_action(0)
    while not world_sarsa.is_done:
        reward = world_sarsa.move(action)
        steps += 1
        if not world_sarsa.is_done:
            new_action = world_sarsa.get_action(0)
        else:
            new_action = np.random.choice([0,1,2,3])
        action = new_action
    if steps < min_steps[0]:
        min_steps[0] = steps
    mean_steps[0] = (i/(i+1))*mean_steps[0] + (1/(i+1))*steps
for i in range(1000):    
    steps = 0
    world_QL.init_position()
    trace.append(np.copy(world_QL.position))
    action = world_QL.get_action(0)
    while not world_QL.is_done:
        reward = world_QL.move(action)
        steps += 1
        if not world_QL.is_done:
            new_action = world_QL.get_action(0)
        else:
            new_action = np.random.choice([0,1,2,3])
        action = new_action
    if steps < min_steps[1]:
        min_steps[1] = steps
    mean_steps[1] = (i/(i+1))*mean_steps[1] + (1/(i+1))*steps
for i in range(1000):    
    steps = 0
    world_ex.init_position()
    trace.append(np.copy(world_ex.position))
    action = world_ex.get_action(0)
    while not world_ex.is_done:
        reward = world_ex.move(action)
        steps += 1
        if not world_ex.is_done:
            new_action = world_ex.get_action(0)
        else:
            new_action = np.random.choice([0,1,2,3])
        action = new_action
    if steps < min_steps[2]:
        min_steps[2] = steps
    mean_steps[2] = (i/(i+1))*mean_steps[2] + (1/(i+1))*steps
    
print(f'convergence timestep: {convergence_time_step[0]}\t{convergence_time_step[1]}\t{convergence_time_step[2]}')
print(f'min steps: {min_steps[0]}\t{min_steps[1]}\t{min_steps[2]}')
print(f'mean steps: {mean_steps[0]}\t{mean_steps[1]}\t{mean_steps[2]}')