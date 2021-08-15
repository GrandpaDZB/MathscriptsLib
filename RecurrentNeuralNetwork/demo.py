

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# plot training parameters
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

# obtain imdb_reviews dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
# obtain the encoder of this dataset
encoder = info.features['text'].encoder

print("Preprocessing data")
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



class SimpleRNN(tf.keras.Model):
    # units is the node_num of a hidden layer
    # h = (batch_size * vector_len)*(vector_len * units) = (batch_size * units)
    def __init__(self, units, layer_num):
        # use Parent class's __init__
        super(SimpleRNN, self).__init__()
        
        self.layer_num = layer_num
        self.embeding_layer = layers.Embedding(encoder.vocab_size , 80, input_length=64)
        self.states = []
        self.outs = []
        self.rnn_layers = []
        for i in range(layer_num):
            self.states.append([tf.zeros([BATCH_SIZE, units])])
            self.rnn_layers.append(layers.SimpleRNNCell(units))
            self.outs.append(0)
        self.out_layer = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=None):
        x = self.embeding_layer(inputs)
        for each in tf.unstack(x, axis=1):
            for i in range(self.layer_num):
                self.outs[i], self.states[i] = self.rnn_layers[i](each, self.states[0])

        x = self.out_layer(self.outs[-1]) 
        return x   

    
        
print("building model")

model = keras.Sequential([
    layers.Embedding(encoder.vocab_size , 64, input_length=64),
    layers.SimpleRNN(64, dropout = 0.2, return_sequences= True),
    layers.SimpleRNN(64, dropout = 0.2, return_sequences= True),
    layers.SimpleRNN(64, dropout = 0.2),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation = 'sigmoid')
])
model.summary()

# model = SimpleRNN(64, 3)
model.compile(
    optimizer= keras.optimizers.Adam(1e-4),
    loss = tf.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy']
)
history = model.fit(
    train_dataset, 
    epochs=10,
    validation_data= test_dataset  
)

def predict(text):
    global encode
    global model
    encode_text = [encoder.encode(text)]
    encode_text = keras.preprocessing.sequence.pad_sequences(encode_text, 64)
    return model.predict(encode_text)
