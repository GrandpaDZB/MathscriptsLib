from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.ops.gen_array_ops import deep_copy

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
    keras.layers.Dense(24, activation='relu'),
    # keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1)
])

model(np.ones([1,5]))
# model = keras.models.load_model("cartpole_model_with_TD(lambda)_60000steps.h5")
optimizer = keras.optimizers.RMSprop(learning_rate = 0.0005)
model.summary()


programing_times = 64
replay_memory_length = 10000
replay_memory = []
gamma = 0.7
tao = 0.2
Epsilon = 0.8
Epsilon_decay_rate = 0.005
Epsilon_min = 0.01


old_theta = []
for each in model.trainable_variables:
    old_theta.append(deep_copy(each))

max_episode_num = 100
for episode in range(max_episode_num):

    print(f'Episode: {episode+1}')
    state = env.reset()
    step = 0

    while True:

        action = greedy_action(model, state, Epsilon)
        if Epsilon > Epsilon_min:
            Epsilon = Epsilon*(1-Epsilon_decay_rate)
        else:
            Epsilon = Epsilon_min
        state_next, reward, done, _ = env.step(action)
        if done and step < 300:
            reward = -10
        step += 1

        # store transition (St, At, R, St+1, done) in replay_memory
        if len(replay_memory) == replay_memory_length:
            replay_memory = replay_memory[1:]
        replay_memory.append([np.copy(state), action, reward, np.copy(state_next), done])

        # Sample and replay
        if len(replay_memory) < programing_times:
            if done:
                break
            continue
        y = []
        e = []
        theta = model.get_weights()
        model.set_weights(old_theta)
        for replay in range(programing_times):
            num_t = np.random.randint(len(replay_memory))
            transition = replay_memory[num_t]
            e.append(transition)

            if transition[-1]:
                Q_expect = transition[2]
            else:
                max_Q_0 = model(np.reshape(np.hstack([transition[3], 0]), (1,5)))
                max_Q_1 = model(np.reshape(np.hstack([transition[3], 1]), (1,5)))
                if max_Q_1 > max_Q_0:
                    Q_expect = transition[2] + gamma*max_Q_1
                else:
                    Q_expect = transition[2] + gamma*max_Q_0
            if type(Q_expect) == int:
                y.append(Q_expect)
            else:
                y.append(Q_expect.numpy()[0,0])
        y = tf.convert_to_tensor(y)

        model.set_weights(theta)

        # ====================================================== GD
        with tf.GradientTape() as tape:
            loss = 0
            for i in range(len(y)):
                loss = (i/(i+1))*loss + (1/(i+1))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))

        grad = tape.gradient(loss, model.trainable_variables)
        # for i in range(len(grad)):
        #     grad[i] = -grad[i]
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # print(f'=>loss: {loss}')
        # with tf.GradientTape() as tape:
        #     loss = 0
        #     for i in range(len(y)):
        #         loss = (i/(i+1))*loss + (1/(i+1))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))

        # grad = tape.gradient(loss, model.trainable_variables)
        # # for i in range(len(grad)):
        # #     grad[i] = -grad[i]
        # optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # print(f'=>loss: {loss}')
        # with tf.GradientTape() as tape:
        #     loss = 0
        #     for i in range(len(y)):
        #         loss = (i/(i+1))*loss + (1/(i+1))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))

        # grad = tape.gradient(loss, model.trainable_variables)
        # # for i in range(len(grad)):
        # #     grad[i] = -grad[i]
        # optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # print(f'=>loss: {loss}')
        # with tf.GradientTape() as tape:
        #     loss = 0
        #     for i in range(len(y)):
        #         loss = (i/(i+1))*loss + (1/(i+1))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))

        # grad = tape.gradient(loss, model.trainable_variables)
        # # for i in range(len(grad)):
        # #     grad[i] = -grad[i]
        # optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # print(f'=>loss: {loss}')
        # with tf.GradientTape() as tape:
        #     loss = 0
        #     for i in range(len(y)):
        #         loss = (i/(i+1))*loss + (1/(i+1))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))*(y[i]-model(np.reshape(np.hstack([e[i][0], e[i][1]]), (1,5))))

        # grad = tape.gradient(loss, model.trainable_variables)
        # # for i in range(len(grad)):
        # #     grad[i] = -grad[i]
        # optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # print(f'=>loss: {loss}')
        # ============================================================== END GD
        
        
        for i in range(len(old_theta)):
            old_theta[i] = tao*old_theta[i] + (1-tao)*model.get_weights()[i]
        # print(f'loss: {loss}')

        if done:
            print(f'step: {step}')
            print(f'loss: {loss}')
            print(f'epsilon: {Epsilon}')
            break
                

model.save("test.h5")

x = np.linspace(-2.4, 2.4, 50)
y = np.linspace(-0.208, 0.208, 50)
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




