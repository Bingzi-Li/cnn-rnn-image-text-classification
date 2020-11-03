import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import csv
import re
import pylab
import os
import pickle
import time

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
HIDDEN_SIZE = 20

batch_size = 128
one_hot_size = 256
no_epochs = 2
lr = 0.01

use_dropout = False
optimizer_ = 'Adam'
name = 'Char_RNN'

# types are: Vanilla, LSTM, 2Layers, GradClipping
improve_type = 'LSTM'

seed = 10
tf.random.set_seed(seed)

# This is required when using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Read data with [character]
def vocabulary(strings):
    chars = sorted(list(set(list(''.join(strings)))))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    return vocab_size, char_to_ix


def preprocess(strings, char_to_ix, MAX_LENGTH):
    data_chars = [list(d.lower()) for _, d in enumerate(strings)]
    for i, d in enumerate(data_chars):
        if len(d) > MAX_LENGTH:
            d = d[:MAX_LENGTH]
        elif len(d) < MAX_LENGTH:
            d += [' '] * (MAX_LENGTH - len(d))

    data_ids = np.zeros([len(data_chars), MAX_LENGTH], dtype=np.int64)
    for i in range(len(data_chars)):
        for j in range(MAX_LENGTH):
            data_ids[i, j] = char_to_ix[data_chars[i][j]]
    return np.array(data_ids)


def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []
    cop = re.compile("[^a-z^A-Z^0-9^,^.^' ']")
    with open('./data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            data = cop.sub("", row[1])
            x_train.append(data)
            y_train.append(int(row[0]))

    with open('./data/test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            data = cop.sub("", row[1])
            x_test.append(data)
            y_test.append(int(row[0]))

    vocab_size, char_to_ix = vocabulary(x_train+x_test)
    x_train = preprocess(x_train, char_to_ix, MAX_DOCUMENT_LENGTH)
    y_train = np.array(y_train)
    x_test = preprocess(x_test, char_to_ix, MAX_DOCUMENT_LENGTH)
    y_test = np.array(y_test)

    x_train = tf.constant(x_train, dtype=tf.int64)
    y_train = tf.constant(y_train, dtype=tf.int64)
    x_test = tf.constant(x_test, dtype=tf.int64)
    y_test = tf.constant(y_test, dtype=tf.int64)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = read_data_chars()
# Use `tf.data` to batch and shuffle the dataset:
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(batch_size)

# Build model
tf.keras.backend.set_floatx('float32')


class CharCNN(Model):
    def __init__(self, vocab_size=256):
        super(CharCNN, self).__init__()
        self.vocab_size = vocab_size
        # Weight variables and RNN cell
        self.conv1 = layers.Conv2D(
            N_FILTERS, FILTER_SHAPE1, padding='VALID', activation='relu', use_bias=True)
        self.pool1 = layers.MaxPool2D(
            POOLING_WINDOW, POOLING_STRIDE, padding='SAME')
        self.conv2 = layers.Conv2D(
            N_FILTERS, FILTER_SHAPE2, padding='VALID', activation='relu', use_bias=True)
        self.pool2 = layers.MaxPool2D(
            POOLING_WINDOW, POOLING_STRIDE, padding='SAME')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        # forward
        x = tf.one_hot(x, one_hot_size)
        x = x[..., tf.newaxis]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class CharRNN(Model):
    def __init__(self, vocab_size=256, hidden_dim=10):
        super(CharRNN, self).__init__()
        # Hyperparameters
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # Weight variables and RNN cell
        if improve_type == 'Vanilla':
            self.rnn = layers.RNN(
                tf.keras.layers.SimpleRNNCell(self.hidden_dim), unroll=True)
        elif improve_type == 'LSTM':
            self.rnn = layers.RNN(
                tf.keras.layers.LSTMCell(self.hidden_dim), unroll=True)
        else:
            self.rnn = layers.RNN(
                tf.keras.layers.GRUCell(self.hidden_dim), unroll=True)
        if improve_type == '2Layers':
            self.rnn2 = layers.RNN(
                tf.keras.layers.GRUCell(self.hidden_dim), unroll=True)
        self.dense = layers.Dense(MAX_LABEL, activation=None)

    def call(self, x, drop_rate):
        # forward logic
        x = tf.one_hot(x, one_hot_size)
        encoding = self.rnn(x)
        if improve_type == '2Layers':
            encoding = self.rnn2(x)
        encoding = tf.nn.dropout(encoding, drop_rate)
        logits = self.dense(encoding)

        return logits


if name == 'Char_CNN':
    model = CharCNN(256)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
elif name == 'Char_RNN':
    model = CharRNN(256, HIDDEN_SIZE)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

# Choose optimizer and loss function for training
if optimizer_ == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
elif optimizer_ == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# Select metrics to measure the loss and the accuracy of the model.
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

# Training function


def train_step(model, x, label, drop_rate):
    with tf.GradientTape() as tape:
        out = model(x, drop_rate)
        loss = loss_object(label, out)
        gradients = tape.gradient(loss, model.trainable_variables)
        if improve_type == 'GradClipping':
            gradients, _ = tf.clip_by_norm(gradients, 2.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, out)

# Testing function


def test_step(model, x, label, drop_rate):
    out = model(x, drop_rate)
    t_loss = loss_object(label, out)
    test_loss(t_loss)
    test_accuracy(label, out)


test_acc = []
train_acc = []
test_cost = []
train_cost = []
time_ = []
for epoch in range(no_epochs):
    start_time = time.time()
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    drop_r = 0.5 if use_dropout else 0.0
    for images, labels in train_ds:
        train_step(model, images, labels, drop_rate=drop_r)

    for images, labels in test_ds:
        test_step(model, images, labels, drop_rate=0)

    test_acc.append(test_accuracy.result())
    train_acc.append(train_accuracy.result())
    test_cost.append(test_loss.result())
    train_cost.append(train_loss.result())
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result(),
                          test_loss.result(),
                          test_accuracy.result()))
    time_.append(time.time() - start_time)
print(f'Avg Time used: {sum(time_)/len(time_)}')

history = dict()
history['train_accuracy'] = train_acc
history['test_accuracy'] = test_acc
history['test_loss'] = test_cost
history['train_loss'] = train_cost

# # save train cost plot
# if not os.path.exists('./results'):
#     os.mkdir('./results')
# pylab.figure()
# pylab.plot(np.arange(no_epochs), history['train_loss'])
# pylab.xlabel('epochs')
# pylab.ylabel('train cost')
# pylab.title(name)
# if use_dropout:
#     pylab.savefig(
#         f'./results/partb1_{name}_train_cost_dropout.pdf')
# else:
#     pylab.savefig(
#         f'./results/partb1_{name}_train_cost_no_dropout.pdf')


# # save test accuracy plot
# pylab.figure()
# pylab.plot(np.arange(no_epochs), history['test_accuracy'])
# pylab.xlabel('epochs')
# pylab.ylabel('test acc')
# pylab.title(name)
# if use_dropout:
#     pylab.savefig(
#         f'./results/partb1_{name}_test_acc_dropout.pdf')
# else:
#     pylab.savefig(
#         f'./results/partb1_{name}_test_acc_no_dropout.pdf')


# save history
if not os.path.exists('./histories'):
    os.mkdir('./histories')
if use_dropout:
    hist_path = f'./histories/partb1_{name}_{improve_type}_dropout'
else:
    hist_path = f'./histories/partb1_{name}_{improve_type}_no_dropout'
with open(hist_path, 'wb') as file_pi:
    pickle.dump(history, file_pi)
