import gym
import tensorflow as tf 
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.engine import training
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def greedy_action(model, state, epsilon = 0.1):
    if np.random.rand() < epsilon:
        action = np.random.choice([0,1])
    else:
        value1 = model(np.reshape(np.hstack([state, 1]), (1,5)))
        value0 = model(np.reshape(np.hstack([state, 0]), (1,5)))
        if value1 >= value0:
            action = 1
        else:
            action = 0
    return action


# gym environment settings
env = gym.make('CartPole-v1')
# array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)
# array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38],dtype=float32)

# neural network
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
model(np.ones([1,5]))
# model = keras.models.load_model("cartpole_model_with_TD(lambda)_60000steps.h5")
weights = model.trainable_variables
optimizer = keras.optimizers.Adam()

# TD-lambda parameters
param_l = len(weights)
z = []
for i in range(param_l):
    z.append(tf.convert_to_tensor(np.zeros(weights[i].shape), dtype='float32'))
gamma = 0.9
lambda_para = 0.75
learning_rate = 0.01

state = env.reset()
action = 0
Q_old = tf.convert_to_tensor(np.zeros((1,1)), dtype='float32')
step = 0
episode_num = 0
good_results = 0
results_counter = 0
value = model(np.reshape(np.hstack([state, action]) , (1,5)))
max_iteration = 20000
for iteration in range(max_iteration):
    # get  observation
    new_state, reward, done, _ = env.step(action)
    step += 1
    if done:
        new_state = env.reset()
        episode_num += 1
        print(f'Episode: {episode_num}\tstep = {step}')
        step = 0
        

    # choose the greedy action
    new_action = greedy_action(model, new_state)
    Q = model(np.reshape(np.hstack([state, action]) , (1,5)))

    if not done:
        Q_new = model(np.reshape(np.hstack([new_state, new_action]) , (1,5)))
    else:
        Q_new = np.zeros((1,1))
    
    # update Eligibility traces
    # compute gradient
    if not done:
        esitimated_value = reward + gamma*model(np.reshape(np.hstack([new_state, new_action]), (1,5)))
    else:
        esitimated_value = reward

    Delta = reward + gamma*Q_new - Q



    with tf.GradientTape() as tape:
        Q = model(np.reshape(np.hstack([state, action]) , (1,5)))
    grads = tape.gradient(Q, model.trainable_variables)

    for i in range(param_l):
        z[i] = gamma*lambda_para*z[i] + grads[i]

    # update weights
    tmp_grads = []
    for i in range(param_l):
        tmp_grads.append(learning_rate*(Delta + Q - Q_old).numpy()*z[i])
        if 1 == tmp_grads[-1].shape[0]:
            tmp_grads[-1] = tmp_grads[-1][0]
    # for i in range(param_l):
    #     weights[i].assign_add(tf.reshape(learning_rate*Delta*z[i], weights[i].shape))
    optimizer.apply_gradients(zip(tmp_grads, model.trainable_variables))

    Q_old = Q_new
    state = new_state
    action = new_action


# ================== TEST

state = env.reset()
step = 0

action = greedy_action(model, state ,epsilon=0)
state, _, done, _ = env.step(action)
step += 1

while not done:
    env.render()
    action = greedy_action(model, state ,epsilon=0)
    state, _, done, _ = env.step(action)
    step += 1
    print(step)
env.close()


# model.save(f'./cartpole_model_with_TD(lambda)_{max_iteration+40000}steps.h5')


x = np.linspace(-4.8, 4.8, 50)
y = np.linspace(-0.418, 0.418, 50)
Q0 = np.zeros((50,50))
Q1 = np.zeros((50,50))
for i in range(50):
    for j in range(50):
        Q0[i,j] = model(np.array([[x[i], 0, y[j], 0, 0]]))
        Q1[i,j] = model(np.array([[x[i], 0, y[j], 0, 1]]))

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.view_init(60, 35)
ax3.set_xlabel('Position')
ax3.set_ylabel('angle')
ax3.set_zlabel('value')
ax3.plot_surface(X,Y,Q1-Q0,cmap='binary')

# fig.savefig(f'{max_iteration+40000}steps.jpg')

