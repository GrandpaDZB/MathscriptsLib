import tensorflow as tf



class Inception(tf.keras.Model):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(c1, kernel_size=1,
                                            activation='relu', padding='same')
        self.conv2_1 = tf.keras.layers.Conv2D(c2[0], kernel_size=1,
                                            activation='relu', padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(c2[1], kernel_size=3,
                                            activation='relu', padding='same')
        self.conv3_1 = tf.keras.layers.Conv2D(c3[0], kernel_size=1,
                                            activation='relu', padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(c3[1], kernel_size=5,
                                            activation='relu', padding='same')
        self.pool4_1 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same',
                                                 strides=1)
        self.conv4_2 = tf.keras.layers.Conv2D(c4, kernel_size=1,
                                            activation='relu', padding='same')
        
            
    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2_2(self.conv2_1(inputs))
        x3 = self.conv3_2(self.conv3_1(inputs))
        x4 = self.conv4_2(self.pool4_1(inputs))
        
        return tf.concat((x1, x2, x3, x4), axis=-1)


def GoogLeNet(input_):
    inputs = tf.keras.layers.Input(shape=input_.shape[1:])
    
    model = [
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                                  activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=1,
                              activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=192, kernel_size=3,
                                  activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D()
    ]
    
    x = inputs
    for layer in model:
        x = layer(x)
        
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
