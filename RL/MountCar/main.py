import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

""" 
Position: [-1.2, 0.6]
Velocity: [-0.07, 0.07]
 """
env = gym.make("MountainCarContinuous-v0")



# ===================================== neural networks

model_action_value = keras.Sequential([
    # keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='relu')
])
model_action_value(np.ones([1,2]))
weights = model_action_value.trainable_variables
optimizer = keras.optimizers.Adam()

def gradient_lnpi(state, action, model):
    with tf.GradientTape() as f, tf.GradientTape() as g:
        f.watch(model.trainable_variables)
        g.watch(model.trainable_variables)
        u = model(state)[0][0]
        v = model(state)[0][1]
    du_theta = f.gradient(u, model.trainable_variables)
    dv_theta = g.gradient(v, model.trainable_variables)
    print(du_theta)
    print(dv_theta)
    

    tmp_1 = (action-u)/(v)*du_theta[0]+((action-u)*(action-u)/v/v/v-1/v)*dv_theta[0]
    tmp_2 = (action-u)/(v)*du_theta[1]+((action-u)*(action-u)/v/v/v-1/v)*dv_theta[1]

    gradient = [tmp_1, tmp_2]
    return gradient