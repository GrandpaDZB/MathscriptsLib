import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Activation, Flatten

def AlexNet(input_):
    inputs = tf.keras.layers.Input(shape=input_.shape[1:])

    model = [
        layers.Conv2D(48, kernel_size=7, strides=2, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # layers.BatchNormalization(),
        # layers.Activation('relu'),

        # layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        # layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # layers.BatchNormalization(),
        # layers.Activation('relu'),

        # layers.Conv2D(192, kernel_size=3, strides=1, padding='same', activation='relu'),

        # layers.Conv2D(192, kernel_size=3, strides=1, padding='same', activation='relu'),

        # layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
        # layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # #layers.Activation('relu'),

        tf.keras.layers.GlobalAvgPool2D(),

        layers.Dense(2048, activation='relu'),
        #layers.Dropout(activations='relu'),

        layers.Dense(2048, activation='relu'),
        #layers.Dropout(activations='relu'),
    ]
    x = inputs
    for layer in model:
        x = layer(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)