import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
import xlrd
import numpy as np

data = xlrd.open_workbook("./data_play.xls")
table = data.sheet_by_index(0)
X = []
for i in range(16):
    X.append(table.col_values(i)[:-1])
Y = table.row_values(4)
X_train = np.array(X)
Y_train = np.array(Y)


model = keras.Sequential()
#model.add(keras.layers.InputLayer(input_shape=(4,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid', name='output'))


Y = model(tf.ones((1,4)))

model.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'],
)

history = model.fit(
    X_train,
    Y_train,
    batch_size=1,
    epochs=1000,
)