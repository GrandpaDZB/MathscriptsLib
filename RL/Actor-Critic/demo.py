import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

last_state = np.zeros((1,210,160,3))
last_done = False

# actions space [0,5]
# observation space (210,160,3)
env = gym.make("Qbert-v0")
env_test = gym.make("Qbert-v0")
action_space_size = 6
gamma = 0.9

def preprocess(state):
    state = np.reshape(state, (1,210,160,3))
    state = state.astype(float)
    return state

# ================================= neural network
class Bottleneck(keras.layers.Layer):
    def __init__(self, output_channel, stride):
        super(Bottleneck, self).__init__()
        self.conv_11_64 = keras.layers.Conv2D(64, 1, (1,1))
        self.BN1 = keras.layers.BatchNormalization()
        self.conv_33 = keras.layers.Conv2D(output_channel, 3, (stride, stride))
        self.BN2 = keras.layers.BatchNormalization()
        self.conv_11_out = keras.layers.Conv2D(output_channel, 1, (1,1))
        self.BN3 = keras.layers.BatchNormalization()
        self.conv_11_res = keras.layers.Conv2D(output_channel, 3, (stride,stride))
        self.BN4 = keras.layers.BatchNormalization()
    def call(self, inputs):
        x = self.conv_11_64(inputs)
        x = self.BN1(x)
        x = keras.activations.relu(x)

        x = self.conv_33(x)
        x = self.BN2(x)
        x = keras.activations.relu(x)

        x = self.conv_11_out(x)
        x = self.BN3(x)
        x = keras.activations.relu(x)

        y = self.conv_11_res(inputs)
        y = self.BN4(y)
        return keras.activations.relu(tf.add(x,y))

class MobileNet(keras.Model):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.L1 = keras.layers.Conv2D(32, 3, (2,2))
        self.L2 = Bottleneck(16, 2)
        self.L3 = Bottleneck(16, 2)
        self.L4 = keras.layers.Conv2D(320, 3, (2,2))
        self.L5 = keras.layers.Conv2D(320, 3, (2,2))  
        self.L6 = keras.layers.AveragePooling2D((5,4))
        self.L7 = keras.layers.Dense(32, activation='relu')
        self.L8 = keras.layers.Dense(32, activation='relu')
        self.L9 = keras.layers.Dense(7)
    def call(self, inputs):
        x = self.L1(inputs)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.L7(x)
        x = self.L8(x)
        x = self.L9(x)
        return tf.reshape(x, (x.shape[0],7))
AC = MobileNet()
AC(np.zeros((1,210,160,3)))
optimizer1 = keras.optimizers.Adam(0.00001)
optimizer2 = keras.optimizers.Adam(0.0001)

 
# ==================================== sample 
def nparray(array, shape):
    l = np.array(array)
    return np.reshape(l, shape)
def choose_action(states):
    global env
    global AC
    policy = tf.nn.softmax(AC(states)[:, 0:action_space_size])
    rand = np.random.rand()
    sum_p = 0
    for i in range(action_space_size):
        sum_p += policy[0,i]
        if rand < sum_p:
            return i
def sample_by_timesteps(time_steps, policy_weight = None):
    global AC
    global env
    global last_state
    global last_done
    if policy_weight != None:
        preserved_weights = AC.get_weights()
        AC.set_weights(policy_weight)
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
            tmp = np.zeros((1,action_space_size))
            tmp[0,action] = 1.0
            actions.append(tmp)
            rewards.append(reward)
        else:
            if len(rewards) != 0:
                rewards[-1] = 0
            rewards.append(0)
            actions.append(np.zeros((1,6)))
            states.append(state)
            dones.append(step)
            done = False
            state = preprocess(env.reset())
        if step == time_steps-1:
                last_done = done
                last_state = state
    return [nparray(states, (time_steps, 210, 160, 3)), nparray(actions, (time_steps, action_space_size)), np.array(rewards), (dones)]

def test_ave_steps(episode, max_step = 100):
    score = 0
    ave_score = 0
    for i in range(episode):
        state = preprocess(env_test.reset())
        done = False
        while not done:
            action = choose_action(state)
            state, reward, done, _ = env_test.step(action)
            score += reward
            state = preprocess(state)
        ave_score = (i/(i+1))*ave_score + (1/(i+1))*score
    return ave_score


# ========================================== accumulated rewards

def compute_accumulated_rewards(rewards, dones, last_state):
    global gamma
    global AC
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
        accumulated_rewards[-1] = AC(np.reshape(last_state, (1,210,160,3)))[0,action_space_size].numpy()
        for i in range(len(rewards)-2, dones[-1], -1):
            accumulated_rewards[i] = gamma*accumulated_rewards[i+1] + rewards[i]
    if dones == []:
        accumulated_rewards[-1] = AC(np.reshape(last_state, (1,210,160,3)))[0,action_space_size].numpy()
        for i in range(len(rewards)-2, -1, -1):
            accumulated_rewards[i] = gamma*accumulated_rewards[i+1] + rewards[i]
    return np.reshape(accumulated_rewards, (len(rewards),1))


# ====================================== optimize Critic
def optimize_Critic(accumulated_rewards, states):
    global AC
    global optimizer1
    with tf.GradientTape() as tape:
        err = keras.losses.mean_squared_error(tf.reshape(accumulated_rewards, (states.shape[0],)),AC(states)[:,action_space_size])
        loss = err
    gradient = tape.gradient(loss, AC.trainable_variables)
    # print(f'Critic loss: {loss}')
    optimizer1.apply_gradients(zip(gradient, AC.trainable_variables))
    return

# ====================================== Advantage function
def estimate_advantage_function(states, rewards, dones):
    global AC
    A = np.zeros((len(rewards),))
    V = AC(states)[:,action_space_size]
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


# ============================================== optimize Actor

def optimize_Actor(actions, states, Advan):
    global AC
    global optimizer
    with tf.GradientTape() as tape:
        logits = AC(states)[:, 0:action_space_size]
        negative_likelyhood = tf.nn.softmax_cross_entropy_with_logits(actions, logits)
        weighted_likelyhood = tf.multiply(negative_likelyhood, Advan)
        loss = tf.reduce_mean(weighted_likelyhood)
    gradients = tape.gradient(loss, AC.trainable_variables)
    optimizer2.apply_gradients(zip(gradients, AC.trainable_variables))
    return

for iteration in range(300):
    states, actions, rewards, dones = sample_by_timesteps(50)
    accumulated_rewards = compute_accumulated_rewards(rewards, dones, states[-1,:,:,:])
    for _ in range(50):
        optimize_Critic(accumulated_rewards,states)
    Advan = estimate_advantage_function(states, rewards, dones)
    for _ in range(1):
        optimize_Actor(actions, states, Advan)
    
    print(f'Iter: {iteration+1}')
    print(AC(states)[0:4,6].numpy())
    print(np.reshape(accumulated_rewards[0:4],(4,)))
    # print(f'Averages step: {step}')
    if iteration%10 == 0:
        step = test_ave_steps(1)
        print(f'Average score: {step}')