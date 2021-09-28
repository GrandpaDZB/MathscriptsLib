import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.pooling import Pooling1D

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
# 每五个数据一提取，对应时间上就是每个小时一个采样点
df = df[5::6]
# 将时间单独提取出来
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

# 给出了数据集的均值/数量/方差等等基本信息
df.describe()


# 数据预处理，除去异常数据
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0
max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()


# 特征工程
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


# 分离数据集，按照 0.7 0.2 0.1 分割训练集，验证集，测试集
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

# 正规化数据，除去量纲的影响
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std



# 定义生成器，input_width是输入时间序列长度， label_width是单位时间长度，shift是预测时间长度
class WindowGenerator():
      def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

            
        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])
        def split_window(self, features):
            inputs = features[:, self.input_slice, :]
            labels = features[:, self.labels_slice, :]
            if self.label_columns is not None:
                labels = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                    axis=-1)
            # Slicing doesn't preserve static shape information, so set the shapes
            # manually. This way the `tf.data.Datasets` are easier to inspect.
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])

            return inputs, labels
        WindowGenerator.split_window = split_window
        def make_dataset(self, data):
            data = np.array(data, dtype=np.float32)
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,)

            ds = ds.map(self.split_window)

            return ds
        WindowGenerator.make_dataset = make_dataset

        @property
        def train(self):
            return self.make_dataset(self.train_df)

        @property
        def val(self):
            return self.make_dataset(self.val_df)

        @property
        def test(self):
            return self.make_dataset(self.test_df)

        @property
        def example(self):
            """Get and cache an example batch of `inputs, labels` for plotting."""
            result = getattr(self, '_example', None)
            if result is None:
                # No example batch was found, so get one from the `.train` dataset
                result = next(iter(self.train))
                # And cache it for next time
                self._example = result
            return result

        def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
            inputs, labels = self.example
            plt.figure(figsize=(12, 8))
            plot_col_index = self.column_indices[plot_col]
            max_n = min(max_subplots, len(inputs))
            for n in range(max_n):
                plt.subplot(max_n, 1, n+1)
                plt.ylabel(f'{plot_col} [normed]')
                plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                        label='Inputs', marker='.', zorder=-10)

                if self.label_columns:
                    label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                    label_col_index = plot_col_index

                if label_col_index is None:
                    continue

                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

                if n == 0:
                    plt.legend()

            plt.xlabel('Time [h]')
        WindowGenerator.plot = plot
        WindowGenerator.train = train
        WindowGenerator.val = val
        WindowGenerator.test = test
        WindowGenerator.example = example

OUT_STEPS = 10
# 定义输入24小时的数据，返回24小时的预测
w1 = WindowGenerator(input_width=24, label_width=24, shift=OUT_STEPS,
                     label_columns=['T (degC)'])
# 定义输入6小时的数据，返回1小时的预测
w2 = WindowGenerator(input_width=4, label_width=1, shift=1,
                     label_columns=['T (degC)'])
# 分离器的不同定义会改变神经网络的结构，举个例子，采用w2的定义方式，那么一次会推给神经网络7个数据
# 使用split_window函数进行分割，获得6个时间单位的输入数据，和1个时间单位的label数据进行训练


single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'])

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(w1.train, epochs=MAX_EPOCHS,
                      validation_data=w1.val,
                      callbacks=[early_stopping])
  return history


# dense = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=1)
# ])

# history = compile_and_fit(dense, w1)

# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.Conv1D(filters=4,
#                            kernel_size=(4,),
#                            activation='relu'),
#     # tf.keras.layers.MaxPool1D(3),
#     tf.keras.layers.Dense(units=24, activation='relu'),
#     # tf.keras.layers.GRU(32, return_sequences=True),
#     # tf.keras.layers.GRU(64, return_sequences=True),
#     # tf.keras.layers.GRU(128, return_sequences=True),
#     # tf.keras.layers.GRU(64, return_sequences=True),
#     # tf.keras.layers.GRU(32, return_sequences=True),
#     # tf.keras.layers.GRU(32, return_sequences=True),
#     # tf.keras.layers.GRU(32, return_sequences=True),
#     # tf.keras.layers.GRU(32, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=1)
# ])

CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(24*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([24, num_features]),
    

    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.GRU(32, return_sequences=True),

    tf.keras.layers.Dense(units=10),

    tf.keras.layers.Dense(units=1),
    
])
history = compile_and_fit(multi_conv_model, w1)


val_performance = {}
performance = {}