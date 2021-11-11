import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


# * ================================================= Game world 

class World:
    def __init__(self):
        self.velocity = 0.4;
        self.wind_strength = 0.4;
        pass

    def is_target_position(self, x, y):
        if x > 6.0 and x < 7.0 and y < 4.5 and y > 3.5:
            return True
        else:
            return False 

    def update(self, state, action):
        # std update
        if action == 0:
            state += np.array([1,0])*self.velocity
            pass
        elif action == 1:
            state += np.array([0.707,0.707])*self.velocity
            pass
        elif action == 2:
            state += np.array([0,1])*self.velocity
            pass
        elif action == 3:
            state += np.array([-0.707,0.707])*self.velocity
            pass
        elif action == 4:
            state += np.array([-1,0])*self.velocity
            pass
        elif action == 5:
            state += np.array([-0.707,-0.707])*self.velocity
            pass
        elif action == 6:
            state += np.array([0,-1])*self.velocity
            pass
        elif action == 7:
            state += np.array([0.707,-0.707])*self.velocity
            pass

        # wind affection
        if state[0][0] > 5:
            state[0][1] += np.random.rand()*self.wind_strength

        # falling detection
        if state[0][0] < 0 or state[0][0] > 10 or state[0][1] < 0 or state[0][1] > 5:
            state = np.zeros((1,2))

        # rewarding     
        if self.is_target_position(state[0][0], state[0][1]):
            reward = 5
            state = np.zeros((1,2))
        else:
            reward = -1
        return [state, reward]

# * ============================================= epsilon greedy strategy

def epsilon_greedy_strategy(model, state):
    epsilon = 0.1
    max_value = -100000
    max_action = 0
    if np.random.rand() < epsilon:
        return np.random.choice(range(8))
    for i in range(8):
        input_vec = np.array([np.hstack([state[0],i])])
        value = model.predict(input_vec)
        # print(value)
        if value[0] > max_value:
            max_value = value[0]
            max_action = i
    return max_action


# * ============================================== neural network
model  = keras.Sequential(
    [
        keras.layers.Dense(16, activation="relu", name="layer1"),
        # keras.layers.Dense(32, activation="relu", name="layer2"),
        # keras.layers.Dense(32, activation="relu", name="layer3"),
        keras.layers.Dense(16, activation="relu", name="layer4"),
        keras.layers.Dense(1,name="output")
    ]
)
x = np.ones((1,3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss = tf.keras.losses.mean_squared_error,
    # metrics=['accuracy']
)
model(x)
model.summary()


# * ============================================= iteration
max_iteration = 5000
n = 20
tao = 0
beta = 0.07

game = World()

state = np.zeros((1,2))
action = 1

n_states = []
n_actions = []
n_rewards = []

mean_reward = 0
updated_state = np.zeros((1,2))
updated_action = 1

for iteration in range(max_iteration):
    [new_state, reward] = game.update(state, action)
    new_action = epsilon_greedy_strategy(model, new_state)
    if iteration - n + 2 >= 0:
        n_rewards.append(reward)
        n_states.append(new_state)
        n_actions.append(action)
        if len(n_rewards) > n:
            updated_state = n_states[0]
            updated_action = n_actions[0]
            n_rewards = n_rewards[1:]
            n_states = n_states[1:]
            n_actions = n_actions[1:]

    tao = iteration - n + 1
    if tao >= 0:
        input_vec = np.array([np.hstack([updated_state[0], updated_action])])
        this_value = model.predict(input_vec)
        delta = np.sum(np.array(n_rewards)-mean_reward) 
        + model.predict(input_vec)
        mean_reward += beta*(delta - this_value)

        model.fit(input_vec, np.ones((1,1))*delta)
    state = new_state
    action = new_action
    print(state)
    print(f'{iteration*100/max_iteration}%')



# * =========================================== test


state = np.ones((1,2))
loci = [state]

timestep = 0
while not game.is_target_position(state[0][0], state[0][1]):
    timestep += 1
    action = epsilon_greedy_strategy(model, state)
    [state, _] = game.update(state, action)
    loci.append(state)
    print(state)
    print(f'timestep: {timestep}')


