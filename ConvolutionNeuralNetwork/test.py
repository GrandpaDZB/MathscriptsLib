import tensorflow as tf
import cv2
import numpy as np
import Resnet as RM
from tensorflow.keras import datasets, layers

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)




X = tf.random.uniform(shape=(1,32,32,3))
model = RM.ResNet(X)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.load_weights("./model/ResNet")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("Finish initialization")

def load_data(path = './1.png'):
    file = path
    img = cv2.imread(file)
    img = cv2.resize(img, (32,32))
    data = np.array([img])
    return data

def get_result(path = './1.png'):
    global model
    global class_names
    data = load_data(path)
    result = model.predict(data)
    print("recognized result: "+class_names[np.argmax(result)])
    print(f'possibility: {result}')

