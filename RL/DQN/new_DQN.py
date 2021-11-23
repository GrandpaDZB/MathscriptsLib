import gym
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle as pkl

max_score = 0
max_episode = 300

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.8

alpha = 0.01 # learning rate
alpha_decay = 0.001

batch_size = 32


memory = deque(maxlen=10000)
env = gym.make('CartPole-v1')
env._max_episode_steps = 1000


model = keras.Sequential(
    [
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(48, activation='relu'),
        keras.layers.Dense(96, activation='relu'),
        keras.layers.Dense(96, activation='relu'),
        keras.layers.Dense(48, activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(2)
    ]
)
model.compile(
    loss = keras.losses.MSE,
    # optimizer = keras.optimizers.Adam(alpha)
    optimizer = keras.optimizers.Adam(alpha, decay = alpha_decay)
    
)
model(np.zeros((1,4)))
model.summary()
# model = keras.models.load_model("./test.h5")

weight_t = model.get_weights()
weight = model.get_weights()

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def choose_action(state, epsilon):
    if np.random.random() <= epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(state))

def get_epsilon(t):
    return max(epsilon_min, min(epsilon,1.0 - math.log10((t+1)*epsilon_decay)))

def preprocess(state):
    return np.reshape(state, (1,4))

def replay(batch_size, epsilon):
    global weight_t
    global weight
    global model
    x_batch = []
    y_batch = []
    minibatch = random.sample(memory, min(len(memory), batch_size))
    weight = model.get_weights()
    model.set_weights(weight_t)
    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma+np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])
    model.set_weights(weight)
    model.fit(
        np.array(x_batch),
        np.array(y_batch),
        batch_size=len(x_batch)
    )
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

history = []
def run():
    global history
    global max_score
    global weight
    global weight_t
    scores = deque(maxlen=100)
    save_counter = 0
    chg_counter = 0
    for e in range(max_episode):
        save_counter += 1
        chg_counter += 1
        state = preprocess(env.reset())
        done = False
        i = 0
        while not done:
            action = choose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            replay(batch_size, get_epsilon(e))
            i += 1
        print(f'Episode = {e+1}\tStep = {i}\tEpsilon = {get_epsilon(e)}')
        history.append(i)
        with open("./history.pkl",  'wb') as f:
            pkl.dump(history, f)
        
        if chg_counter > 0:
            weight_t = weight
            chg_counter = 0
        if save_counter > 20:
            model.save(f'./test.h5')
            save_counter = 0
        
run()


state = env.reset()
done = False
state = preprocess(state)
while not done:
    action = choose_action(state, 0)
    state, reward, done, _ = env.step(action)
    env.render()
    state = preprocess(state)
env.close()
