import tensorflow as tf
from tensorflow import keras
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import gym

gamma = 0.9
last_state = np.zeros((1,4))
last_done = False


env = gym.make('CartPole-v1')
env_test = gym.make('CartPole-v1')
env._max_episode_steps = 1000
env_test._max_episode_steps = 10000
# the outputs are [logits, V]
Actor_Critic = keras.Sequential([
    keras.layers.Dense(32, activation="relu"),
    # keras.layers.Dense(64, activation="relu"),
    # keras.layers.Dense(128, activation="relu"),
    # keras.layers.Dense(128, activation="relu"),
    # keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(3)
])
Actor_Critic(np.random.random((1,4)))
# 0.008/0.008 for 32/32/3, gamma = 0.9
# 0.002/0.001 for 32/64/128/128/64/32/3 gamma = 0.9
optimizer1 = keras.optimizers.Adam(learning_rate=0.008)
optimizer2 = keras.optimizers.Adam(learning_rate=0.008)

# action - (N*T) * Da tensor of actions
# states - (N*T) * Ds tensor of states
# A_values - (N*T) * 1 tensor of estimated advantage function
def optimize_Actor(actions, states, Advan):
    global Actor_Critic
    global optimizer
    with tf.GradientTape() as tape:
        logits = Actor_Critic(states)[:, 0:2]
        negative_likelyhood = tf.nn.softmax_cross_entropy_with_logits(actions, logits)
        weighted_likelyhood = tf.multiply(negative_likelyhood, Advan)
        loss = tf.reduce_mean(weighted_likelyhood)
    gradients = tape.gradient(loss, Actor_Critic.trainable_variables)
    optimizer2.apply_gradients(zip(gradients, Actor_Critic.trainable_variables))
    return
    
# states - (N*T) * Ds tensor of states
# accumulated_rewards - (N*T) * 1 tensor of total rewards
def optimize_Critic(accumulated_rewards, states):
    global Actor_Critic
    global optimizer
    with tf.GradientTape() as tape:
        err = keras.losses.mean_squared_error(tf.reshape(accumulated_rewards, (states.shape[0],)),Actor_Critic(states)[:,2])
        loss = err
    gradient = tape.gradient(loss, Actor_Critic.trainable_variables)
    # print(f'Critic loss: {loss}')
    optimizer1.apply_gradients(zip(gradient, Actor_Critic.trainable_variables))
    return

def preprocess(state):
    return np.resize(state, (1,4))
def nparray(array, shape):
    l = np.array(array)
    return np.reshape(l, shape)

def choose_action(states):
    global env
    global Actor_Critic
    policy = tf.nn.softmax(Actor_Critic(states)[:, 0:2])
    if np.random.rand() < policy[0,0]:
        return 0
    else:
        return 1


def sample_by_timesteps(time_steps, policy_weight = None):
    global Actor_Critic
    global env
    global last_state
    global last_done
    if policy_weight != None:
        preserved_weights = Actor_Critic.get_weights()
        Actor_Critic.set_weights(policy_weight)
    states = []
    actions = []
    rewards = []
    dones = []

    if last_state.any() == False:
        state = preprocess(env.reset())
    else: 
        state = last_state
    done = last_done
    for step in range(time_steps):
        if not done:
            states.append(state)
            action = choose_action(state)
            state, reward, done, _ = env.step(action)
            state = preprocess(state)
            if action == 0:
                actions.append([1,0])
            else:
                actions.append([0,1])
            rewards.append(reward)
        else:
            if len(rewards) != 0:
                rewards[-1] = 0
            rewards.append(0)
            actions.append([0,0])
            states.append(state)
            dones.append(step)
            done = False
            state = preprocess(env.reset())
        if step == time_steps-1:
                last_done = done
                last_state = state
    return [nparray(states, (time_steps, 4)), nparray(actions, (time_steps, 2)), np.array(rewards), (dones)]

def compute_accumulated_rewards(rewards, dones, last_state):
    global gamma
    global Actor_Critic
    accumulated_rewards = np.zeros((len(rewards),))
    for t in range(len(dones)):
        if t != 0:
            start_point = dones[t-1] + 1
        else:
            start_point = 0
        end_point = dones[t]
        for i in range(0,end_point-start_point+1):
            if i == 0:
                accumulated_rewards[end_point] = 0
            else:
                accumulated_rewards[end_point-i] = accumulated_rewards[end_point+1-i]*gamma + rewards[end_point-i]
    if dones != [] and dones[-1] != len(rewards)-1:
        accumulated_rewards[-1] = Actor_Critic.predict(np.reshape(last_state, (1,4)))[0,2]
        for i in range(len(rewards)-2, dones[-1], -1):
            accumulated_rewards[i] = gamma*accumulated_rewards[i+1] + rewards[i]
    return np.reshape(accumulated_rewards, (len(rewards),1))

def estimate_advantage_function(states, rewards, dones):
    global Actor_Critic
    A = np.zeros((len(rewards),))
    V = Actor_Critic(states)[:,2]
    for i in range(len(A)):
        if i in dones:
            A[i] = 0
        elif i+1 in dones:
            A[i] = -V[i]
        elif i == len(A)-1:
            A[i] = rewards[i]
        else:
            A[i] = rewards[i] + V[i+1] - V[i]
    return A

def test_ave_steps(episode):
    step = 0
    ave_step = 0
    for i in range(episode):
        state = preprocess(env_test.reset())
        done = False
        while not done:
            step += 1
            action = choose_action(state)
            state, _, done, _ = env_test.step(action)
            state = preprocess(state)
        ave_step = (i/(i+1))*ave_step + (1/(i+1))*step
        step = 0
    return ave_step

def test_with_plot(use_10k):
    w = Actor_Critic.get_weights()
    if use_10k:
        Actor_Critic.load_weights("./10k_weights.h5")
    else:
        Actor_Critic.load_weights("./tmp_weights.h5")
    env_plot = gym.make('CartPole-v1')
    env_plot._max_episode_steps = 10000
    state = preprocess(env_plot.reset())
    done = False
    while not done:
        env_plot.render()
        action = choose_action(state)
        state, _, done, _ = env_plot.step(action)
        state = preprocess(state)
    Actor_Critic.set_weights(w)
    env_plot.close()
    
    
max_episode = 150
save_counter = 0

history = []
for iteration in range(3000):
    states, actions, rewards, dones = sample_by_timesteps(200)
    accumulated_rewards = compute_accumulated_rewards(rewards, dones, states[-1])
    for _ in range(20):
        optimize_Critic(accumulated_rewards,states)
    Advan = estimate_advantage_function(states, rewards, dones)
    for _ in range(3):
        optimize_Actor(actions, states, Advan)
    step = test_ave_steps(2)
    print(f'Iter: {iteration+1}')
    print(f'Averages step: {step}')
    history.append(step)
    
    if step >= 100:
        Actor_Critic.save_weights("./tmp_weights.h5")
    if iteration >= 20 and history[-3] - history[-1] > 100 and history[-3]-history[-2] > 100 :
        Actor_Critic.load_weights("./tmp_weights.h5")
    if iteration >= 100 and history[-1] < 20:
        Actor_Critic.load_weights("./tmp_weights.h5")
    if step == 10000:
        Actor_Critic.save_weights("./10k_weights.h5")
    with open("./history.pkl", "wb") as f:
        pkl.dump(history, f)

x = np.linspace(-2.4, 2.4, 50)
y = np.linspace(-0.2, 0.2, 50)
Q0 = np.zeros((50,50,2))
for i in range(50):
    for j in range(50):
        Q0[i,j] = tf.nn.softmax(Actor_Critic(np.array([[x[i], 0, y[j], 0]]))[:,0:2])
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.view_init(60, 35)
ax3.set_xlabel('Position')
ax3.set_ylabel('angle')
ax3.set_zlabel('value')
ax3.plot_surface(X,Y,Q0[:,:,1]-Q0[:,:,0],cmap='binary')

states, actions, rewards, dones = sample_by_timesteps(200)
accumulated_rewards = compute_accumulated_rewards(rewards, dones, states[-1])
for _ in range(100):
    optimize_Critic(accumulated_rewards, states)
Advan = estimate_advantage_function(states, rewards, dones)


