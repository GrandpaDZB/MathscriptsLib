import tensorflow as tf
from tensorflow import keras



'''
brief: 自定义神经网络层，继承自keras.layers.Layer类 
param {*} units=32, input_dim=32
return {*}
'''
class myDense_1(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(myDense_1,self).__init__()

        # 建立可训练参数w，b
        self.w = self.add_weight(
            shape=(input_dim, units), 
            initializer=keras.initializers.random_normal, 
            trainable=True
        )
        self.b = self.add_weight(
            shape=(1, units),
            initializer=keras.initializers.zeros,
            trainable=True
        )

    # call 函数用于定义层的输出
    def call(self, inputs):
        return tf.matmul(inputs, self.w)+self.b


'''
brief: 对于更多的情况而言，层在被创建时只能知道需要多少个unit，而不知道输入的dim，可以在build方法中延迟创建参数的大小，
build方法将在call方法首次调用时自动调用，来创建参数
param {*} units=32
return {*}
'''
class myDense_2(keras.layers.Layer):
    def __init__(self, units=32):
        super(myDense_2,self).__init__()
        self.units= units

    def build(self, input_shape):
        # 建立可训练参数w，b
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), 
            initializer=keras.initializers.random_normal, 
            trainable=True
        )
        self.b = self.add_weight(
            shape=(1, self.units),
            initializer=keras.initializers.zeros,
            trainable=True
        )

    # call 函数用于定义层的输出
    def call(self, inputs):
        return tf.matmul(inputs, self.w)+self.b


'''
brief: 当有了层，可能就会需要块，来更加简洁地表达重复的单元，即直接建立了层的组合，
下面的块就使用了三个全连接层，用relu进行激活，最终输出一个数
param {*}
return {*}
'''
class myBlock_1(keras.layers.Layer):
    def __init__(self):
        super(myBlock_1, self).__init__()
        self.dense_1 = myDense_2(32)
        self.dense_2 = myDense_2(32)
        self.dense_3 = myDense_2(1)
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = keras.activations.relu(x)
        x = self.dense_2(x)
        x = keras.activations.relu(x)
        return self.dense_3(x)

'''
description: Model和Layer看似是相同的，但是Model开放了训练，对于model类可以直接调用fit，compile等方法
对数据集进行训练，总的来说，由层构成块，由块构成模型
param {*}
return {*}
'''
class myModel_1(keras.Model):
    def __init__(self):
        super(myModel_1, self).__init__()
        self.block_1 = myBlock_1()
        
    def call(self, inputs):
        x = self.block_1(inputs)
        return x



x = tf.ones((4,4))
linear_layer = myBlock_1()
y = linear_layer(x)
print(y)




