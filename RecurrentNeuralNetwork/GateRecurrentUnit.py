
import tensorflow as tf
from tensorflow.python.keras.backend import conv1d
from tensorflow.python.keras.layers.convolutional import Conv1D
tf.compat.v1.enable_eager_execution()
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.recurrent_v2 import GRU
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk


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
    def build(self,input_shape):
        self.input_shap = input_shape

    def call(self, inputs):
        input_size = self.input_shap[-1]
        left_num = (input_size-self.kernel_size)%self.step
        cov_num = int((input_size-self.kernel_size-left_num)/self.step)
        #y = tf.zeros((1,cov_num+1+left_num))
        y = []
        for i in range(int(cov_num)+1):
            y.append(tf.matmul(tf.reshape(self.kernel,(1,5)), tf.reshape(inputs[0][2*i:2*i+5],(5,1))))
        for i in range(left_num):
            y.append(tf.reshape(inputs[0][2*cov_num+5+i],(1,left_num)))
        return tf.concat(y,1)


with open("./train_x.pkl", 'rb') as f:
    X = pk.load(f)
with open("./train_y.pkl", 'rb') as f:
    Y = pk.load(f)
train_x = X[0:100,0:256]
test_x = X[101:,0:256]
train_y = Y[0:100]
test_y = Y[101:]


''' # obtain imdb_reviews dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
# obtain the encoder of this dataset
encoder = info.features['text'].encoder


# preprocess the dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_x = []
train_y = []
test_x = []
test_y = []
for i in train_dataset:
    train_x.append(i[0].numpy())
    train_y.append(i[1].numpy())
for j in test_dataset:
    test_x.append(j[0].numpy())
    test_y.append(j[1].numpy())
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, 64)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, 64)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder = True)



print("building model") '''

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train_dataset = train_dataset.batch(64, drop_remainder=True)
test_dataset = test_dataset.batch(64, drop_remainder = True)



model = keras.Sequential([
    # layers.Embedding(encoder.vocab_size , 64, input_length=64),
    layers.GRU(64, return_sequences=True),
    layers.GRU(128, return_sequences=True),
    layers.GRU(256, return_sequences=True),
    # layers.GRU(128, return_sequences=True),
    # # layers.GRU(64, return_sequences=True),
    # # layers.GRU(64, return_sequences=True),
    # # layers.GRU(64, return_sequences=True),
    # # layers.GRU(64, return_sequences=True),
    # layers.GRU(32, recurrent_dropout= 0.5),
    # # layers.Dense(64, activation='relu'),
    # # layers.Dropout(0.5),
    # cov1d(),
    # cov1d(),
    # cov1d(),
    # cov1d(),
    # layers.Conv1D(16,4,activation='relu'),
    # layers.Conv1D(16,4,activation='relu'),
    # layers.Conv1D(16,4,activation='relu'),
    

    layers.Dense(10, activation = 'relu'),
    layers.Dense(1, activation='sigmoid')
])
y = model(tf.ones((1,256)))
model.summary()



# model = SimpleRNN(64, 3)
model.compile(
    optimizer= keras.optimizers.Adam(1e-3),
    loss = tf.losses.BinaryCrossentropy(),
    metrics = ['accuracy']
)
# history = model.fit(
#     train_dataset, 
#     epochs=10,
#     validation_data= test_dataset  
# )
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs = 100,
)


