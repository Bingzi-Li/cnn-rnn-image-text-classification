import pickle
import os
import pylab
import re
import csv
from tensorflow.keras import Model, layers
import numpy as np
import collections
import tensorflow as tf
from nltk.tokenize import word_tokenize
import time
# import nltk
# nltk.download('punkt')

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
HIDDEN_SIZE = 20
EMBEDDING_SIZE = 20
FILTER_SHAPE1 = [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

batch_size = 128
no_epochs = 200
lr = 0.01

seed = 10
tf.random.set_seed(seed)

use_dropout = False
optimizer_ = 'Adam'
name = 'Word_RNN'

# types are: Vanilla, LSTM, 2Layers, GradClipping
improve_type = 'LSTM'

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


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def build_word_dict(contents):
    words = list()
    for content in contents:
        for word in word_tokenize(clean_str(content)):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<eos>"] = 2
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)
    return word_dict


def preprocess(contents, word_dict, document_max_len):
    x = list(map(lambda d: word_tokenize(clean_str(d)), contents))
    x = list(
        map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d))
                 * [word_dict["<pad>"]], x))
    return x


def read_data_words():
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

    word_dict = build_word_dict(x_train+x_test)
    x_train = preprocess(x_train, word_dict, MAX_DOCUMENT_LENGTH)
    y_train = np.array(y_train)
    x_test = preprocess(x_test, word_dict, MAX_DOCUMENT_LENGTH)
    y_test = np.array(y_test)

    x_train = [x[:MAX_DOCUMENT_LENGTH] for x in x_train]
    x_test = [x[:MAX_DOCUMENT_LENGTH] for x in x_test]
    x_train = tf.constant(x_train, dtype=tf.int64)
    y_train = tf.constant(y_train, dtype=tf.int64)
    x_test = tf.constant(x_test, dtype=tf.int64)
    y_test = tf.constant(y_test, dtype=tf.int64)

    vocab_size = tf.get_static_value(tf.reduce_max(x_train))
    vocab_size = max(vocab_size, tf.get_static_value(
        tf.reduce_max(x_test))) + 1
    return x_train, y_train, x_test, y_test, vocab_size


x_train, y_train, x_test, y_test, vocab_size = read_data_words()
# print(vocab_size)
# Use `tf.data` to batch and shuffle the dataset:
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(batch_size)

# Build model
tf.keras.backend.set_floatx('float32')


class WordCNN(Model):

    def __init__(self, vocab_size):
        super(WordCNN, self).__init__()
        # Hyperparameters
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(
            vocab_size, EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LENGTH)

        # Weight variables and CNN layers
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
        # forward logic
        # print("input shape:", x.shape)
        embedding = self.embedding(x)
        # print("embedding shape:", embedding.shape)
        embedding = embedding[..., tf.newaxis]
        x = self.conv1(embedding)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class WordRNN(Model):

    def __init__(self, vocab_size, hidden_dim=10):
        super(WordRNN, self).__init__()
        # Hyperparameters
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(
            vocab_size, EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LENGTH)
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
        embedding = self.embedding(x)
        encoding = self.rnn(embedding)
        if improve_type == '2Layers':
            encoding = self.rnn2(x)
        encoding = tf.nn.dropout(encoding, drop_rate)
        logits = self.dense(encoding)

        return logits


if name == 'Word_CNN':
    model = WordCNN(vocab_size)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
elif name == 'Word_RNN':
    model = WordRNN(vocab_size, HIDDEN_SIZE)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

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


def test_step(model, x, label, drop_rate=0):
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
    # Reset the metrics at the start of the next epoch
    start_time = time.time()
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
