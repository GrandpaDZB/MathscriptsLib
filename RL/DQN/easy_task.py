from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from tensorflow.python.ops.gen_array_ops import deep_copy

def greedy_action(model, state, epsilon = 0.1):
    if np.random.rand() < epsilon:
        action = np.random.choice([0,1,2])
    else:
        value2 = model(np.reshape(np.hstack([state, 2]), (1,3)))
        value1 = model(np.reshape(np.hstack([state, 1]), (1,3)))
        value0 = model(np.reshape(np.hstack([state, 0]), (1,3)))
        return np.argmax([value0, value1, value2])
    return action


# gym environment settings
env = gym.make('MountainCar-v0')
# array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)
# array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38],dtype=float32)

# neural network
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model(np.ones([1,3]))
# model = keras.models.load_model("cartpole_model_with_TD(lambda)_60000steps.h5")
optimizer = keras.optimizers.RMSprop(learning_rate = 0.01, momentum=0.95, rho=0.95)
model.summary()


programing_times = 10
replay_memory_length = 1000
replay_memory = []
gamma = 0.99

old_theta = []
for each in model.trainable_variables:
    old_theta.append(deep_copy(each))


max_episode_num = 10
for episode in range(max_episode_num):

    print(f'Episode: {episode+1}')
    t = 0
    state = env.reset()
    step = 0
    rewards = 0

    while True:

        action = greedy_action(model, state)
        state_next, reward, done, _ = env.step(action)
        if done and step < 300:
            print(f'done reward: {reward}')
        step += 1
        rewards += reward

        # store transition (St, At, R, St+1, done) in replay_memory
        if len(replay_memory) == replay_memory_length:
            replay_memory = replay_memory[1:]
        replay_memory.append([np.copy(state), action, reward, np.copy(state_next), done])

        transition = replay_memory[-1]
        # if transition[-1]:
        #     Q_expect = transition[2]
        # else:
        #     max_Q_0 = model(np.reshape(np.hstack([transition[3], 0]), (1,3)))
        #     max_Q_1 = model(np.reshape(np.hstack([transition[3], 1]), (1,3)))
        #     max_Q_2 = model(np.reshape(np.hstack([transition[3], 2]), (1,3)))
        #     Q_expect = transition[2] + np.argmax([max_Q_0, max_Q_1, max_Q_2])
        max_Q_0 = model(np.reshape(np.hstack([transition[3], 0]), (1,3)))
        max_Q_1 = model(np.reshape(np.hstack([transition[3], 1]), (1,3)))
        max_Q_2 = model(np.reshape(np.hstack([transition[3], 2]), (1,3)))
        Q_expect = transition[2] + np.argmax([max_Q_0, max_Q_1, max_Q_2])

        with tf.GradientTape() as tape:
            Q = model(np.reshape(np.hstack([transition[0], transition[1]]), (1,3)))
            loss = keras.losses.mean_squared_error(Q_expect, Q)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        # Sample and replay
        if len(replay_memory) < programing_times:
            continue
        for replay in range(programing_times):
            num_t = np.random.randint(len(replay_memory))
            transition = replay_memory[num_t]
            # if transition[-1]:
            #     Q_expect = transition[2]
            # else:
            #     max_Q_0 = model(np.reshape(np.hstack([transition[3], 0]), (1,3)))
            #     max_Q_1 = model(np.reshape(np.hstack([transition[3], 1]), (1,3)))
            #     max_Q_2 = model(np.reshape(np.hstack([transition[3], 2]), (1,3)))
            #     Q_expect = transition[2] + np.argmax([max_Q_0, max_Q_1, max_Q_2])
            max_Q_0 = model(np.reshape(np.hstack([transition[3], 0]), (1,3)))
            max_Q_1 = model(np.reshape(np.hstack([transition[3], 1]), (1,3)))
            max_Q_2 = model(np.reshape(np.hstack([transition[3], 2]), (1,3)))
            Q_expect = transition[2] + np.argmax([max_Q_0, max_Q_1, max_Q_2])

            with tf.GradientTape() as tape:
                Q = model(np.reshape(np.hstack([transition[0], transition[1]]), (1,3)))
                loss = keras.losses.mean_squared_error(Q_expect, Q)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
        
        if done:
            print(f'loss: {loss}')
            print(f'step: {step}')
            break
                
            
model.save(f'./MountainCar_test.h5')



x = np.linspace(-1.2, 0.6, 50)
y = np.linspace(-0.07, 0.07, 50)
#Q0 = np.zeros((50,50))
Q2 = np.zeros((50,50))
for i in range(50):
    for j in range(50):
        #Q0[i,j] = model(np.array([[x[i], 0, y[j], 0, 0]]))
        Q2[i,j] = model(np.array([[x[i], y[j], 2]]))

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.view_init(60, 35)
ax3.set_xlabel('Position')
ax3.set_ylabel('angle')
ax3.set_zlabel('value')
ax3.plot_surface(X,Y,Q2,cmap='binary')

