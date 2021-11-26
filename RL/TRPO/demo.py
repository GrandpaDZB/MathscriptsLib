from itertools import accumulate
from numpy.lib.function_base import gradient
from numpy.random.mtrand import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle as pkl
import gym
from tensorflow.python.keras.engine.base_layer import Layer

env = gym.make('CartPole-v1')
env_test = gym.make('CartPole-v1')

gamma = 0.9

last_state = np.zeros((1,4))
last_done = False

Actor = keras.Sequential([
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(2)
])
Actor(np.zeros((1,4)))

Layers = []
Layers.append(keras.layers.Dense(24, activation='relu'))
# Layers.append(keras.layers.Dense(48, activation='relu'))
# Layers.append(keras.layers.Dense(48, activation='relu'))from itertools import accumulate
from numpy.lib.function_base import gradient
from numpy.random.mtrand import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle as pkl
import gym
from tensorflow.python.keras.engine.base_layer import Layer

env = gym.make('CartPole-v1')
env_test = gym.make('CartPole-v1')

gamma = 0.9

last_state = np.zeros((1,4))
last_done = False

Actor = keras.Sequential([
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(2)
])
Actor(np.zeros((1,4)))

Layers = []
Layers.append(keras.layers.Dense(24, activation='relu'))
# Layers.append(keras.layers.Dense(48, activation='relu'))
# Layers.append(keras.layers.Dense(48, activation='relu'))
Layers.append(keras.layers.Dense(24, activation='relu'))
Layers.append(keras.layers.Dense(2))
input_data = np.zeros((1,4))
for each in Layers:
    input_data = each(input_data)
length_layers = len(Layers)


Critic = keras.Sequential([
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1)
])
Critic(np.zeros((1,4)))
optimizer_Critic = keras.optimizers.Adam(0.005, decay = 0.01)

def optimize_Critic(accumulated_rewards, states):
    global Critic
    global optimizer_Critic
    with tf.GradientTape() as tape:
        err = keras.losses.MSE(accumulated_rewards,Critic(states))
        loss = tf.reduce_mean(err)
    gradient = tape.gradient(loss, Critic.trainable_variables)
    optimizer_Critic.apply_gradients(zip(gradient, Critic.trainable_variables))
    return

def Actor_predict(state):
    global Layers
    global length_layers
    for l in range(length_layers):
        if l == 0:
            out = Layers[l](state)
        else:
            out = Layers[l](out)
    return out

def choose_action(state):
    out = Actor_predict(state)
    policy = tf.nn.softmax(out)
    if np.random.rand() <=policy[0,0]:
        return 0
    else:
        return 1

def preprocess(state):
    return np.resize(state, (1,4))

def nparray(array, shape):
    l = np.array(array)
    return np.reshape(l, shape)
    
def compute_accumulated_rewards(rewards, dones, last_state):
    global gamma
    global Critic
    accumulated_rewards = np.zeros((len(rewards),))

    for t in range(len(dones)):
        if t != 0:
            start_point = dones[t-1]
        else:
            start_point = 0
        end_point = dones[t]


        for i in range(0,end_point-start_point):
            if i == 0:
                accumulated_rewards[end_point] = 0
            else:
                accumulated_rewards[end_point-i] = accumulated_rewards[end_point+1-i]*gamma + rewards[end_point-i]
    if dones[-1] != len(rewards)-1:
        accumulated_rewards[-1] = Critic.predict(np.reshape(last_state, (1,4)))[0,0]
        for i in range(len(rewards)-2, dones[-1], -1):
            accumulated_rewards[i] = gamma*accumulated_rewards[i+1] + rewards[i]
    
    return np.reshape(accumulated_rewards, (len(rewards),1))



def sample_an_episode(policy_weight = None):
    if policy_weight != None:
        preserved_weights = Actor.get_weights()
        Actor.set_weights(policy_weight)
    states = []
    actions = []
    rewards = []
    batch_size = 0
    
    state = preprocess(env_test.reset())
    done = False
    step = 0
    while not done:
        step += 1
        batch_size += 1
        states.append(state)
        action = choose_action(state)
        state, reward, done, _ = env_test.step(action)
        state = preprocess(state)
        if action == 0:
            actions.append([1,0])
        else:
            actions.append([0,1])
        rewards.append(reward)
    states.append(state)
    rewards[-1] = 0
    if policy_weight != None:
        Actor.set_weights(preserved_weights)
    return [nparray(states,(batch_size+1,4)), nparray(actions, (batch_size,2)), np.array(rewards), step]

def sample_by_timesteps(time_steps, policy_weight = None):
    global Actor
    global env
    global last_state
    global last_done
    if policy_weight != None:
        preserved_weights = Actor.get_weights()
        Actor.set_weights(policy_weight)
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
    global Critic
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
        accumulated_rewards[-1] = Critic.predict(np.reshape(last_state, (1,4)))[0,0]
        for i in range(len(rewards)-2, dones[-1], -1):
            accumulated_rewards[i] = gamma*accumulated_rewards[i+1] + rewards[i]
    return np.reshape(accumulated_rewards, (len(rewards),1))

def estimate_advantage_function(states, rewards, dones):
    global Critic
    A = np.zeros((len(rewards),))
    V = Critic(states)
    for i in range(len(A)):
        if i in dones:
            A[i] = 0
        elif i+1 in dones:
            A[i] = -V[i,0]
        elif i == len(A)-1:
            A[i] = rewards[i]
        else:
            A[i] = rewards[i] + V[i+1,0] - V[i,0]
    return A

def merge_params(params):
    merged_params = []
    for i in range(int(len(params)/2)):
        merged_params.append(tf.concat([params[2*i], tf.reshape(params[2*i+1], (1,params[2*i+1].shape[0]))], 0))
    return merged_params
def demerge_params(params):
    demerged_params = []
    for i in range(len(params)):
        demerged_params.append(params[i][:-1,:])
        demerged_params.append(params[i][-1,:])
    return demerged_params


        
def compute_policy_gradient_and_Fisher_Information_Matrix(states, actions, A):
    global Actor
    with tf.GradientTape() as tape1:
        lnpi = -tf.nn.softmax_cross_entropy_with_logits(actions, Actor(states[:-1,:]))
        F = [0 for x in range(len(Actor.layers))]
        for i in range(len(actions)):
            with tf.GradientTape() as tmp_tape:
                each_lnpi = -tf.nn.softmax_cross_entropy_with_logits(actions, Actor(states[:-1,:]))[i]
            g = merge_params(tmp_tape.gradient(each_lnpi, Actor.trainable_variables))
            for j in range(len(g)):
                F[j] = (i/(i+1))*F[j] + (1/(i+1))*tf.matmul(g[j], tf.transpose(g[j]))
        weighted_lnpi = tf.multiply(lnpi, A)
    gradient = tape1.gradient(weighted_lnpi, Actor.trainable_variables)
    return [merge_params(gradient), F]

def concat_cols(M):
    (m,n) = M.shape
    new_M = M[:,0]
    for i in range(n-1):
        new_M = tf.concat([new_M, M[:,i+1]], 0)
    return np.reshape(new_M, (m*n, 1))
def deconcat_cols(M, n):
    new_M = M[0:n]
    M = M[n:]
    while M.shape[0] >= n:
        new_M = tf.concat([new_M, M[0:n]], 1)
        M = M[n:]
    return new_M


def conjugate_Gradient(A, b, tolerance = 0.0001):
    (m,n) = b.shape
    One = np.eye(n)
    x = np.zeros((n*m,1))
    b = concat_cols(b)
    A = np.kron(One, A)
    r = b-A@x
    P = r
    while True:
        a = (r.T@r)/(P.T@A@P)
        x = x + a*P
        new_r = r - a*A@P
        if (sum(abs(new_r))) < tolerance:
            break
        beta = (new_r.T@new_r)/(r.T@r)
        P = new_r + beta*P
        r = new_r
    return deconcat_cols(x, m)

def compute_natural_gradient_step(F, g, TrustRegion_delta = 0.1, use_conjugate_gradient = True):
    Delta = []
    for i in range(len(g)):
        if tf.linalg.matrix_rank(F[i]) != F[i].shape[0]:
            F[i] += np.eye(F[i].shape[0])*0.1
            print("F is invertable")
            # Delta.append(0*g[i])
            # continue
        if use_conjugate_gradient:
            x = conjugate_Gradient(F[i], g[i])
        else:
            x = tf.matmul(tf.linalg.inv(F[i]),g[i])
        tmp = (concat_cols(g[i]).T@concat_cols(x))
        if tmp <= 0:
            Delta.append(0*x)
        else:
            Delta.append(np.sqrt(2*TrustRegion_delta/(concat_cols(g[i]).T@concat_cols(x)))*x)
    return demerge_params(Delta)
        
def estimate_objective_function(policy_weights=None, episode = 10):
    J = 0
    ave_step = 0
    for each in range(episode):
        states, actions, rewards, step = sample_an_episode(policy_weights)
        accumulated_rewards = compute_accumulated_rewards(rewards)
        A = estimate_advantage_function(states, rewards, [])
        J = (each/(each+1))*J + (1/(each+1))*np.mean(A)
        ave_step = (each/(each+1))*ave_step + (1/(each+1))*step
    return J,step

# def optimize_Actor(states, actions, A):
#     global Actor
#     print('Computing Policy Gradient and Fisher Information Matrix')
#     g, F = compute_policy_gradient_and_Fisher_Information_Matrix(states, actions, A)
#     print('Computing Natural Gradient')
#     Delta = compute_natural_gradient_step(F, g, 10.0, False)
#     weights = Actor.get_weights()
#     weights_bkp = Actor.get_weights()

#     alpha = 0.5
#     decay_time = 0
#     old_J = 0
#     new_J = -1
#     for _ in range(50):
#         weights = weights_bkp
#         if new_J > old_J:
#             break
#         for i in range(len(Delta)):
#             weights[i] += np.math.pow(alpha, decay_time)*Delta[i]
#             decay_time += 1
#         old_J,_ = estimate_objective_function()
#         new_J,_ = estimate_objective_function(weights)
#     else:
#         weights = weights_bkp
#     Actor.set_weights(weights)


def optimize_Actor(x, g):
    global Layers
    delta = 1E-14
    max_eta = 1E-5
    # delta = 1E-17
    # max_eta = 1E-8
    for l in range(length_layers):
        coeff = np.reshape(x[l].T, (1,x[l].shape[0]*x[l].shape[1]))@np.reshape(g[l].numpy().T, (g[l].shape[0]*g[l].shape[1],1))
        adaptive_eta = np.math.sqrt(2*delta/(abs(coeff[0,0])+1E-16))
        eta = min(adaptive_eta, max_eta)
        print(eta)
        if eta == np.nan:
            eta = 0
        x[l] = eta*x[l]
    x = demerge_params(x)
    for l in range(length_layers):
        weights = Layers[l].get_weights()
        weights[0] -= x[2*l]
        weights[1] -= x[2*l+1]
        Layers[l].set_weights(weights)

def train_one_episode():
    print("Sampling for an episode")
    states, actions, rewards, step = sample_an_episode()
    print(f'step: {step}')
    print('Computing accumulated rewards')
    accumulated_rewards = compute_accumulated_rewards(rewards)
    print('Optimizing Critic')
    optimize_Critic(accumulated_rewards, states)
    print('Estimating Advantage function')
    A = estimate_advantage_function(states, rewards, [])
    print('Optimizing Actor')
    optimize_Actor(states, actions, A)


def KF_approximation_for_F(states, actions):
    global Layers
    F = [0 for _ in range(length_layers)]
    S = [0 for _ in range(length_layers)]
    A = [0 for _ in range(length_layers)]

    input_datas = states
    for i in range(len(actions)):
        input_data = tf.reshape(input_datas[i,:], (1,4))
        # 迭代每一个层，求解Gradient(L,s)
        for l in range(len(F)):
            # input_data = tf.concat([input_data, np.ones((input_data.shape[0], 1))], 1)
            a = tf.concat([input_data, np.array([[1]])], 1)
            A[l] = (i/(i+1))*A[l] + (1/(i+1))*tf.matmul(tf.transpose(a), a)
            with tf.GradientTape() as tape:
                input_data = Layers[l](input_data)
                tape.watch(input_data)
                tmp = input_data
                for k in range(l+1, length_layers):
                    tmp = Layers[k](tmp)
                L = -tf.nn.softmax_cross_entropy_with_logits(actions[i], tmp)
            grad_L_s = tape.gradient(L, input_data)
            S[l] = (i/(i+1))*S[l] + (1/(i+1))*tf.matmul(tf.transpose(grad_L_s), grad_L_s)
            pass
    for i in range(length_layers):
        F[i] = np.kron(A[i], S[i])
    return [F,S,A]

def KF_approximation_for_Delta(states, actions, Advan):
    global Layers
    F = [0 for _ in range(length_layers)]
    S = [0 for _ in range(length_layers)]
    A = [0 for _ in range(length_layers)]

    input_datas = states
    for i in range(len(actions)):
        input_data = tf.reshape(input_datas[i,:], (1,4))
        # 迭代每一个层，求解Gradient(L,s)
        for l in range(len(F)):
            # input_data = tf.concat([input_data, np.ones((input_data.shape[0], 1))], 1)
            a = tf.concat([input_data, np.array([[1]])], 1)
            A[l] = (i/(i+1))*A[l] + (1/(i+1))*tf.matmul(tf.transpose(a), a)
            with tf.GradientTape() as tape:
                input_data = Layers[l](input_data)
                tape.watch(input_data)
                tmp = input_data
                for k in range(l+1, length_layers):
                    tmp = Layers[k](tmp)
                L = -tf.nn.softmax_cross_entropy_with_logits(actions[i], tmp)
            grad_L_s = tape.gradient(L, input_data)
            S[l] = (i/(i+1))*S[l] + (1/(i+1))*tf.matmul(tf.transpose(grad_L_s), grad_L_s)
            pass

    input_datas = states
    with tf.GradientTape(persistent=True) as tape:
        # weights = []
        # for l in range(length_layers):    
        #     weights.append(Layers[l].trainable_variables)
        # tape.watch(weights)
        for l in range(length_layers):
            input_datas = Layers[l](input_datas)
        lnpi = -tf.nn.softmax_cross_entropy_with_logits(actions, input_datas)
        loss = tf.multiply(lnpi, Advan)
        loss = tf.reduce_mean(loss)
    gradients = []
    for i in range(length_layers):
        gradients.append(tape.gradient(loss, Layers[i].trainable_variables))
    g = []
    for each in gradients:
        g.append(each[0])
        g.append(each[1])
    g = merge_params(g)
        

    inv_A = []
    inv_S = []
    for each_S in S:
        try:
            inv_S.append(np.linalg.inv(each_S))
        except:
            print("Matrix in S has no inverse!")
            inv_S.append(np.linalg.inv(each_S+1E-5*np.eye(each_S.shape[0])))
    for each_A in A:
        try:
            inv_A.append(np.linalg.inv(each_A))
        except:
            print("Matrix in A has no inverse!")
            inv_A.append(np.linalg.inv(each_A+1E-5*np.eye(each_A.shape[0])))
    x = []
    for i in range(length_layers):
        x.append(inv_A[i]@g[i].numpy()@inv_S[i])    
    return [x,g]
        

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

def train_with_timesteps_samples(timesteps = 150):
    states, actions, rewards, dones = sample_by_timesteps(timesteps)
    accumulated_rewards = compute_accumulated_rewards(rewards, dones, states[-1])
    for _ in range(20):
        optimize_Critic(accumulated_rewards, states)
    Advan = estimate_advantage_function(states, rewards, dones)
    x,g = KF_approximation_for_Delta(states, actions, Advan)
    optimize_Actor(x, g)
    steps = test_ave_steps(2)
    print(f'After training, sample runs for {steps} steps')



for _ in range(500):
    train_with_timesteps_samples()

x = np.linspace(-2.4, 2.4, 50)
y = np.linspace(-0.2, 0.2, 50)
Q0 = np.zeros((50,50,2))
for i in range(50):
    for j in range(50):
        Q0[i,j] = tf.nn.softmax(Actor_predict(np.array([[x[i], 0, y[j], 0]])))

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
x,g = KF_approximation_for_Delta(states, actions, Advan)


# direct = []
# for i in range(len(Advan)):
#     if Advan[i] > 0:
#         direct.append(actions[i,:])
#     else:
#         direct.append(np.abs(actions[i,:]-1))
# policy_old = tf.nn.softmax(states)
# optimize_Actor(x,g)
# policy_new = tf.nn.softmax(states)

# real_direct = []
# for i in range(len(Advan)):
#     if policy_new[i,0]-policy_old[i,0] > 0:
#         real_direct.append(np.array([1,0]))
#     else:
#         real_direct.append(np.array([0,1]))
# same = []
# for i in range(len(Advan)):
#     if real_direct[i][0] == direct[i][0]:
#         same.append(True)
#     else:
#         same.append(False)

Layers.append(keras.layers.Dense(24, activation='relu'))
Layers.append(keras.layers.Dense(2))
input_data = np.zeros((1,4))
for each in Layers:
    input_data = each(input_data)
length_layers = len(Layers)


Critic = keras.Sequential([
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1)
])
Critic(np.zeros((1,4)))
optimizer_Critic = keras.optimizers.Adam(0.005, decay = 0.01)

def optimize_Critic(accumulated_rewards, states):
    global Critic
    global optimizer_Critic
    with tf.GradientTape() as tape:
        err = keras.losses.MSE(accumulated_rewards,Critic(states))
        loss = tf.reduce_mean(err)
    gradient = tape.gradient(loss, Critic.trainable_variables)
    optimizer_Critic.apply_gradients(zip(gradient, Critic.trainable_variables))
    return

def Actor_predict(state):
    global Layers
    global length_layers
    for l in range(length_layers):
        if l == 0:
            out = Layers[l](state)
        else:
            out = Layers[l](out)
    return out

def choose_action(state):
    out = Actor_predict(state)
    policy = tf.nn.softmax(out)
    if np.random.rand() <=policy[0,0]:
        return 0
    else:
        return 1

def preprocess(state):
    return np.resize(state, (1,4))

def nparray(array, shape):
    l = np.array(array)
    return np.reshape(l, shape)
    
def compute_accumulated_rewards(rewards, dones, last_state):
    global gamma
    global Critic
    accumulated_rewards = np.zeros((len(rewards),))

    for t in range(len(dones)):
        if t != 0:
            start_point = dones[t-1]
        else:
            start_point = 0
        end_point = dones[t]


        for i in range(0,end_point-start_point):
            if i == 0:
                accumulated_rewards[end_point] = 0
            else:
                accumulated_rewards[end_point-i] = accumulated_rewards[end_point+1-i]*gamma + rewards[end_point-i]
    if dones[-1] != len(rewards)-1:
        accumulated_rewards[-1] = Critic.predict(np.reshape(last_state, (1,4)))[0,0]
        for i in range(len(rewards)-2, dones[-1], -1):
            accumulated_rewards[i] = gamma*accumulated_rewards[i+1] + rewards[i]
    
    return np.reshape(accumulated_rewards, (len(rewards),1))



def sample_an_episode(policy_weight = None):
    if policy_weight != None:
        preserved_weights = Actor.get_weights()
        Actor.set_weights(policy_weight)
    states = []
    actions = []
    rewards = []
    batch_size = 0
    
    state = preprocess(env_test.reset())
    done = False
    step = 0
    while not done:
        step += 1
        batch_size += 1
        states.append(state)
        action = choose_action(state)
        state, reward, done, _ = env_test.step(action)
        state = preprocess(state)
        if action == 0:
            actions.append([1,0])
        else:
            actions.append([0,1])
        rewards.append(reward)
    states.append(state)
    rewards[-1] = 0
    if policy_weight != None:
        Actor.set_weights(preserved_weights)
    return [nparray(states,(batch_size+1,4)), nparray(actions, (batch_size,2)), np.array(rewards), step]

def sample_by_timesteps(time_steps, policy_weight = None):
    global Actor
    global env
    global last_state
    global last_done
    if policy_weight != None:
        preserved_weights = Actor.get_weights()
        Actor.set_weights(policy_weight)
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
    global Critic
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
        accumulated_rewards[-1] = Critic.predict(np.reshape(last_state, (1,4)))[0,0]
        for i in range(len(rewards)-2, dones[-1], -1):
            accumulated_rewards[i] = gamma*accumulated_rewards[i+1] + rewards[i]
    return np.reshape(accumulated_rewards, (len(rewards),1))

def estimate_advantage_function(states, rewards, dones):
    global Critic
    A = np.zeros((len(rewards),))
    V = Critic(states)
    for i in range(len(A)):
        if i in dones:
            A[i] = 0
        elif i+1 in dones:
            A[i] = -V[i,0]
        elif i == len(A)-1:
            A[i] = rewards[i]
        else:
            A[i] = rewards[i] + V[i+1,0] - V[i,0]
    return A

def merge_params(params):
    merged_params = []
    for i in range(int(len(params)/2)):
        merged_params.append(tf.concat([params[2*i], tf.reshape(params[2*i+1], (1,params[2*i+1].shape[0]))], 0))
    return merged_params
def demerge_params(params):
    demerged_params = []
    for i in range(len(params)):
        demerged_params.append(params[i][:-1,:])
        demerged_params.append(params[i][-1,:])
    return demerged_params


        
def compute_policy_gradient_and_Fisher_Information_Matrix(states, actions, A):
    global Actor
    with tf.GradientTape() as tape1:
        lnpi = -tf.nn.softmax_cross_entropy_with_logits(actions, Actor(states[:-1,:]))
        F = [0 for x in range(len(Actor.layers))]
        for i in range(len(actions)):
            with tf.GradientTape() as tmp_tape:
                each_lnpi = -tf.nn.softmax_cross_entropy_with_logits(actions, Actor(states[:-1,:]))[i]
            g = merge_params(tmp_tape.gradient(each_lnpi, Actor.trainable_variables))
            for j in range(len(g)):
                F[j] = (i/(i+1))*F[j] + (1/(i+1))*tf.matmul(g[j], tf.transpose(g[j]))
        weighted_lnpi = tf.multiply(lnpi, A)
    gradient = tape1.gradient(weighted_lnpi, Actor.trainable_variables)
    return [merge_params(gradient), F]

def concat_cols(M):
    (m,n) = M.shape
    new_M = M[:,0]
    for i in range(n-1):
        new_M = tf.concat([new_M, M[:,i+1]], 0)
    return np.reshape(new_M, (m*n, 1))
def deconcat_cols(M, n):
    new_M = M[0:n]
    M = M[n:]
    while M.shape[0] >= n:
        new_M = tf.concat([new_M, M[0:n]], 1)
        M = M[n:]
    return new_M


def conjugate_Gradient(A, b, tolerance = 0.0001):
    (m,n) = b.shape
    One = np.eye(n)
    x = np.zeros((n*m,1))
    b = concat_cols(b)
    A = np.kron(One, A)
    r = b-A@x
    P = r
    while True:
        a = (r.T@r)/(P.T@A@P)
        x = x + a*P
        new_r = r - a*A@P
        if (sum(abs(new_r))) < tolerance:
            break
        beta = (new_r.T@new_r)/(r.T@r)
        P = new_r + beta*P
        r = new_r
    return deconcat_cols(x, m)

def compute_natural_gradient_step(F, g, TrustRegion_delta = 0.1, use_conjugate_gradient = True):
    Delta = []
    for i in range(len(g)):
        if tf.linalg.matrix_rank(F[i]) != F[i].shape[0]:
            F[i] += np.eye(F[i].shape[0])*0.1
            print("F is invertable")
            # Delta.append(0*g[i])
            # continue
        if use_conjugate_gradient:
            x = conjugate_Gradient(F[i], g[i])
        else:
            x = tf.matmul(tf.linalg.inv(F[i]),g[i])
        tmp = (concat_cols(g[i]).T@concat_cols(x))
        if tmp <= 0:
            Delta.append(0*x)
        else:
            Delta.append(np.sqrt(2*TrustRegion_delta/(concat_cols(g[i]).T@concat_cols(x)))*x)
    return demerge_params(Delta)
        
def estimate_objective_function(policy_weights=None, episode = 10):
    J = 0
    ave_step = 0
    for each in range(episode):
        states, actions, rewards, step = sample_an_episode(policy_weights)
        accumulated_rewards = compute_accumulated_rewards(rewards)
        A = estimate_advantage_function(states, rewards, [])
        J = (each/(each+1))*J + (1/(each+1))*np.mean(A)
        ave_step = (each/(each+1))*ave_step + (1/(each+1))*step
    return J,step

# def optimize_Actor(states, actions, A):
#     global Actor
#     print('Computing Policy Gradient and Fisher Information Matrix')
#     g, F = compute_policy_gradient_and_Fisher_Information_Matrix(states, actions, A)
#     print('Computing Natural Gradient')
#     Delta = compute_natural_gradient_step(F, g, 10.0, False)
#     weights = Actor.get_weights()
#     weights_bkp = Actor.get_weights()

#     alpha = 0.5
#     decay_time = 0
#     old_J = 0
#     new_J = -1
#     for _ in range(50):
#         weights = weights_bkp
#         if new_J > old_J:
#             break
#         for i in range(len(Delta)):
#             weights[i] += np.math.pow(alpha, decay_time)*Delta[i]
#             decay_time += 1
#         old_J,_ = estimate_objective_function()
#         new_J,_ = estimate_objective_function(weights)
#     else:
#         weights = weights_bkp
#     Actor.set_weights(weights)


def train_one_episode():
    print("Sampling for an episode")
    states, actions, rewards, step = sample_an_episode()
    print(f'step: {step}')
    print('Computing accumulated rewards')
    accumulated_rewards = compute_accumulated_rewards(rewards)
    print('Optimizing Critic')
    optimize_Critic(accumulated_rewards, states)
    print('Estimating Advantage function')
    A = estimate_advantage_function(states, rewards, [])
    print('Optimizing Actor')
    optimize_Actor(states, actions, A)


def KF_approximation_for_F(states, actions):
    global Layers
    F = [0 for _ in range(length_layers)]
    S = [0 for _ in range(length_layers)]
    A = [0 for _ in range(length_layers)]

    input_datas = states
    for i in range(len(actions)):
        input_data = tf.reshape(input_datas[i,:], (1,4))
        # 迭代每一个层，求解Gradient(L,s)
        for l in range(len(F)):
            # input_data = tf.concat([input_data, np.ones((input_data.shape[0], 1))], 1)
            a = tf.concat([input_data, np.array([[1]])], 1)
            A[l] = (i/(i+1))*A[l] + (1/(i+1))*tf.matmul(tf.transpose(a), a)
            with tf.GradientTape() as tape:
                input_data = Layers[l](input_data)
                tape.watch(input_data)
                tmp = input_data
                for k in range(l+1, length_layers):
                    tmp = Layers[k](tmp)
                L = -tf.nn.softmax_cross_entropy_with_logits(actions[i], tmp)
            grad_L_s = tape.gradient(L, input_data)
            S[l] = (i/(i+1))*S[l] + (1/(i+1))*tf.matmul(tf.transpose(grad_L_s), grad_L_s)
            pass
    for i in range(length_layers):
        F[i] = np.kron(A[i], S[i])
    return [F,S,A]

def KF_approximation_for_Delta(states, actions, Advan):
    global Layers
    F = [0 for _ in range(length_layers)]
    S = [0 for _ in range(length_layers)]
    A = [0 for _ in range(length_layers)]

    input_datas = states
    for i in range(len(actions)):
        input_data = tf.reshape(input_datas[i,:], (1,4))
        # 迭代每一个层，求解Gradient(L,s)
        for l in range(len(F)):
            # input_data = tf.concat([input_data, np.ones((input_data.shape[0], 1))], 1)
            a = tf.concat([input_data, np.array([[1]])], 1)
            A[l] = (i/(i+1))*A[l] + (1/(i+1))*tf.matmul(tf.transpose(a), a)
            with tf.GradientTape() as tape:
                input_data = Layers[l](input_data)
                tape.watch(input_data)
                tmp = input_data
                for k in range(l+1, length_layers):
                    tmp = Layers[k](tmp)
                L = -tf.nn.softmax_cross_entropy_with_logits(actions[i], tmp)
            grad_L_s = tape.gradient(L, input_data)
            S[l] = (i/(i+1))*S[l] + (1/(i+1))*tf.matmul(tf.transpose(grad_L_s), grad_L_s)
            pass

    input_datas = states
    with tf.GradientTape(persistent=True) as tape:
        # weights = []
        # for l in range(length_layers):    
        #     weights.append(Layers[l].trainable_variables)
        # tape.watch(weights)
        for l in range(length_layers):
            input_datas = Layers[l](input_datas)
        lnpi = -tf.nn.softmax_cross_entropy_with_logits(actions, input_datas)
        loss = tf.multiply(lnpi, Advan)
        loss = tf.reduce_mean(loss)
    gradients = []
    for i in range(length_layers):
        gradients.append(tape.gradient(loss, Layers[i].trainable_variables))
    g = []
    for each in gradients:
        g.append(each[0])
        g.append(each[1])
    g = merge_params(g)
        

    inv_A = []
    inv_S = []
    for each_S in S:
        try:
            inv_S.append(np.linalg.inv(each_S))
        except:
            print("Matrix in S has no inverse!")
            inv_S.append(np.linalg.inv(each_S+1E-5*np.eye(each_S.shape[0])))
    for each_A in A:
        try:
            inv_A.append(np.linalg.inv(each_A))
        except:
            print("Matrix in A has no inverse!")
            inv_A.append(np.linalg.inv(each_A+1E-5*np.eye(each_A.shape[0])))
    x = []
    for i in range(length_layers):
        x.append(inv_A[i]@g[i].numpy()@inv_S[i])    
    return [x,g]
        

    

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

def train_with_timesteps_samples(timesteps = 150):
    states, actions, rewards, dones = sample_by_timesteps(timesteps)
    accumulated_rewards = compute_accumulated_rewards(rewards, dones, states[-1])
    for _ in range(20):
        optimize_Critic(accumulated_rewards, states)
    Advan = estimate_advantage_function(states, rewards, dones)
    x,g = KF_approximation_for_Delta(states, actions, Advan)
    optimize_Actor(x, g)
    steps = test_ave_steps(2)
    print(f'After training, sample runs for {steps} steps')



for _ in range(500):
    train_with_timesteps_samples()

x = np.linspace(-2.4, 2.4, 50)
y = np.linspace(-0.2, 0.2, 50)
Q0 = np.zeros((50,50,2))
for i in range(50):
    for j in range(50):
        Q0[i,j] = tf.nn.softmax(Actor_predict(np.array([[x[i], 0, y[j], 0]])))

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
x,g = KF_approximation_for_Delta(states, actions, Advan)


# direct = []
# for i in range(len(Advan)):
#     if Advan[i] > 0:
#         direct.append(actions[i,:])
#     else:
#         direct.append(np.abs(actions[i,:]-1))
# policy_old = tf.nn.softmax(states)
# optimize_Actor(x,g)
# policy_new = tf.nn.softmax(states)

# real_direct = []
# for i in range(len(Advan)):
#     if policy_new[i,0]-policy_old[i,0] > 0:
#         real_direct.append(np.array([1,0]))
#     else:
#         real_direct.append(np.array([0,1]))
# same = []
# for i in range(len(Advan)):
#     if real_direct[i][0] == direct[i][0]:
#         same.append(True)
#     else:
#         same.append(False)
