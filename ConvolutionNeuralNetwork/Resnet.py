import tensorflow as tf
from tensorflow.keras import layers


class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            strides=strides, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels,
                                       kernel_size=1,
                                       strides=strides)
        else:
            self.conv3 = None
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        
    def call(self, x):
        y = tf.nn.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return tf.nn.relu(y + x)


class ResnetBlock(tf.keras.Model):
    def __init__(self, num_channels, num_residuals, first_block=False):
        super().__init__()
        self.listLayers=[]
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels,
                                                use_1x1conv=True,
                                                strides=2))
            else:
                self.listLayers.append(Residual(num_channels))
                
    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x

# class ResNet(tf.keras.Model):
#     def __init__(self, num_blocks):
#         super().__init__()
#         self.conv=layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
#         self.bn=layers.BatchNormalization()
#         self.relu=layers.Activation('relu')
        
#         self.mp=layers.MaxPool2D(pool_size=3, strides=2, padding='same')
#         self.resnet_block1=ResnetBlock(64,num_blocks[0], first_block=True)
        
#         self.resnet_block2=ResnetBlock(128,num_blocks[1])
        
#         self.resnet_block3=ResnetBlock(256,num_blocks[2])
        
#         self.resnet_block4=ResnetBlock(512,num_blocks[3])
        
#         self.gap=layers.GlobalAvgPool2D()
#         self.fc=layers.Dense(units=10,activation=tf.keras.activations.softmax)

#     def call(self, x):
#         x=self.conv(x)
#         x=self.bn(x)
#         x=self.relu(x)
#         x=self.mp(x)
#         x=self.resnet_block1(x)
#         x=self.resnet_block2(x)
#         x=self.resnet_block3(x)
#         x=self.resnet_block4(x)
#         x=self.gap(x)
#         x=self.fc(x)
#         return x

def ResNet(input_):
    inputs = tf.keras.layers.Input(shape=input_.shape[1:])

    model = [
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=input_.shape[1:-1]),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        layers.GlobalAvgPool2D(),
    ]
    x = inputs
    for layer in model:
        x = layer(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)