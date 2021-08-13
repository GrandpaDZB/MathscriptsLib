import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dense

# 数据读取
with open("./b_depressed.csv", 'rt') as f:
    data = list(csv.reader(f))[1:]
    for i in range(len(data)):
        try:
            if "" in data[i]:
                data.pop(i)
        except:
            break
    data = np.array(data, dtype='float')
max_array = data.max(0)
min_array = data.min(0)
for i in range(len(data)):
    data[i,:] = (data[i,:]-min_array)/(max_array-min_array)


# 训练集与测试集的划分
N = data.shape[0]
dim = data.shape[1]
X_train = data[0:int(N*0.8), 0:-1]
X_test = data[int(N*0.8):, 0:-1]
Y_train = data[0:int(N*0.8), -1].reshape((1127,1))
Y_test = data[int(N*0.8):, -1].reshape((282,1))

# 建立神经网络模型
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(22)),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01)),
        
        # layers.Dense(16, activation='relu'),
        # layers.Dense(32, activation='relu'),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(256, activation='relu'),
        # layers.Dense(512, activation='relu'),
        # layers.Dense(1024, activation='relu'),
        # layers.Dense(1024, activation='relu'),
        # layers.Dense(512, activation='relu'),
        # layers.Dense(256, activation='relu'),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(32, activation='relu'),

        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ]
)

# model = keras.Sequential(
#     [
#         # layers.Dropout(0.95, seed=tf.random.set_seed(0)),
#         layers.Dense(512, activation='relu'),
#         #layers.Dropout(0.5, seed=tf.random.set_seed(0)),
#         layers.Dense(1024, activation='relu'),
#         #layers.Dropout(0.5, seed=tf.random.set_seed(0)),
#         layers.Dense(1024, activation='relu'),
#         #layers.Dropout(0.5, seed=tf.random.set_seed(0)),
#         layers.Dense(512, activation='relu'),
#         ##layers.Dropout(0.5, seed=tf.random.set_seed(0)),
#         layers.Dense(512, activation='relu'),
#         ##layers.Dropout(0.5, seed=tf.random.set_seed(0)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(1, activation='sigmoid')
#     ]
# )

y = model(np.ones((1,22)))

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(X_train, Y_train, batch_size=200, epochs=200)

test_scores = model.evaluate(X_test, Y_test)
print(test_scores)