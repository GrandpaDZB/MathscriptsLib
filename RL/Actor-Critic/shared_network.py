import tensorflow as tf
from tensorflow import keras
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import gym


epsilon = 1.0
epsilon_decay = 0.99

gamma = 0.9



env = gym.make('CartPole-v1')
env._max_episode_steps = 1000
# the outputs are [logits, V]
Actor_Critic = keras.Sequential([
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(3)
])
Actor_Critic(np.random.random((1,4)))
optimizer = keras.optimizers.Adam(learning_rate=0.1, decay=0.0001)

# action - (N*T) * Da tensor of actions
# states - (N*T) * Ds tensor of states
# A_values - (N*T) * 1 tensor of estimated advantage function
def optimize_Actor(actions, states, A_values):
    global Actor
    global optimizer_Actor
    with tf.GradientTape() as tape:
        logits = Actor_Critic(states)[:, 0:2]
        negative_likelyhood = tf.nn.softmax_cross_entropy_with_logits(actions, logits)
        weighted_likelyhood = tf.multiply(negative_likelyhood, A_values)
        loss = tf.reduce_mean(weighted_likelyhood)
    gradients = tape.gradient(loss, Actor_Critic.trainable_variables)
    print(f'Actor loss: {loss}')
    optimizer.apply_gradients(zip(gradients, Actor_Critic.trainable_variables))
    return
    
# states - (N*T) * Ds tensor of states
# accumulated_rewards - (N*T) * 1 tensor of total rewards
def optimize_Critic(states, accumulated_rewards):
    global batch_size
    global Critic
    global optimize_Critic
    # Critic.fit(states, accumulated_rewards)
    with tf.GradientTape() as tape2:
        old = Actor_Critic(states)[:,-1]
        # new = np.copy(old)
        # new[:, -1] = np.resize(accumulated_rewards, (batch_size))
        # err = new-old
        err = accumulated_rewards-old
        loss = tf.reduce_mean(tf.multiply(err, err))
        # loss = tf.reduce_mean(tf.multiply(err,err))
        # loss = tf.reduce_sum(loss)
        # loss = tf.multiply(err, err)
    gradients = tape2.gradient(loss, Actor_Critic.trainable_variables)

    print(f'Critic loss: {loss}')
    optimizer.apply_gradients(zip(gradients, Actor_Critic.trainable_variables))
    return

def preprocess(state):
    return np.resize(state, (1,4))

def choose_action(state, epsilon):
    global env
    global Actor
    # policy = Actor.predict(state)
    # if np.random.rand() < policy[0,0]:
    #     return 0
    # else:
    #     return 1
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Actor_Critic.predict(state)[0,0:2])

def apply_epsilon_decay():
    global epsilon
    global epsilon_decay
    if epsilon > 0.01:
        epsilon *= epsilon_decay
    else:
        epsilon = 0.01
    return


actions = []
states = []
rewards = []
accumulated_rewards = []
dones = []
A_values = []
def append_batch(action, state, reward, done):
    global actions
    global states
    global rewards
    global dones
    if action == 0:
        actions.append([1,0])
    else:
        actions.append([0,1])
    states.append(state)
    rewards.append(reward)
    dones.append(done)
def clear_batch():
    global actions
    global states
    global rewards
    global dones
    global accumulated_rewards
    global A_values
    actions = []
    states = []
    rewards = []
    accumulated_rewards = []
    dones = []
    A_values = []


max_episode = 150
save_counter = 0
history = []
for episode in range(max_episode):
    save_counter += 1
    print(f'Episode: {episode+1}')
    state = preprocess(env.reset())
    done = False
    step = 0
    clear_batch()
    while not done:
        step += 1
        action = choose_action(state, epsilon)
        apply_epsilon_decay()
        new_state, reward, done, _ = env.step(action)
        if done:
            reward = 0
        append_batch(action, state, reward, done)
        state = preprocess(new_state)
        
        if done:
            batch_size = len(actions)
            accumulated_reward = 0
            accumulated_rewards = np.zeros((batch_size,))
            for i in range(batch_size):
                accumulated_reward = rewards[batch_size-1-i] + gamma*accumulated_reward
                # for j in range(forward_step):
                #     if i+j >= batch_size:
                #         break 
                #     if dones[i+j]:
                #         break
                #     accumulated_reward += np.math.pow(gamma, j)*rewards[i+j]
                #     if j == forward_step-1:
                #         accumulated_reward += (np.math.pow(gamma, j+1)*Critic(states[i+j])).numpy()[0,0]
                accumulated_rewards[batch_size-1-i]=(accumulated_reward)
            states = np.resize(states, (batch_size, 4))
            optimize_Critic(states, accumulated_rewards)
            for i in range(batch_size):
                if dones[i]:
                    A_values.append(0)
                elif i+1 < batch_size:
                    A_values.append(rewards[i]+Actor_Critic.predict(np.resize(states[i+1,:], (1,4)))[0,-1]-Actor_Critic.predict(np.resize(states[i,:], (1,4)))[0,-1])
                else:
                    A_values.append(0)
            actions = np.array(actions)
            A_values = np.array(A_values)
            A_values = np.resize(A_values, (batch_size,))
            optimize_Actor(actions, states, A_values)
    print(f'Step: {step}')
    history.append(step)
    if save_counter >= 20:
        with open("./history.pkl", 'wb') as f:
            pkl.dump(history, f)
        Actor_Critic.save_weights("./Actor_Critic_weights.h5")        

            

print(Actor_Critic.predict(states))
print(accumulated_rewards)
print(tf.nn.softmax((Actor_Critic.predict(states)[:,0:2])))


