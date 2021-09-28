import tensorflow as tf
from tensorflow import keras
import numpy as np

class cov1d(keras.layers.Layer):
    def __init__(self, kernel_size = 5, step = 2):
        super(cov1d, self).__init__()
        self.kernel_size = kernel_size
        self.step = step
        self.kernel=self.add_weight(
            shape=(kernel_size,),
            initializer=keras.initializers.random_normal,
            trainable=True
        )

    def call(self, inputs):
        try:
            input_size = inputs.shape[0]
        except:
            input_size = inputs.size
        left_num = (input_size-self.kernel_size)%self.step
        cov_num = int((input_size-self.kernel_size-left_num)/self.step)
        #y = tf.zeros((1,cov_num+1+left_num))
        y = []
        for i in range(int(cov_num)+1):
            y.append(tf.matmul(tf.reshape(self.kernel,(1,5)), tf.reshape(inputs[2*i:2*i+5],(5,1))))
        for i in range(left_num):
            y.append(tf.reshape(inputs[2*cov_num+5+i],(1,left_num)))
        return tf.concat(y,1)[0]

x = tf.ones((64,))
layer = cov1d()
x = layer(x)
x = layer(x)
#x = keras.layers.Dense(x)
print(x)
