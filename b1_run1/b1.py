import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import csv
import re
import pylab
import os
import pickle

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

batch_size = 128
one_hot_size = 256
no_epochs = 250
lr = 0.01

seed = 10
tf.random.set_seed(seed)

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

    def call(self, x, drop_rate=0.5):
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


# Training function
def train_step(model, x, label, drop_rate):
    with tf.GradientTape() as tape:
        out = model(x, drop_rate)
        loss = loss_object(label, out)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, out)

# Testing function


def test_step(model, x, label, drop_rate=0):
    out = model(x, drop_rate)
    t_loss = loss_object(label, out)
    test_loss(t_loss)
    test_accuracy(label, out)

# save plot


def save_plot(hist, name, type_, use_dropout):
    # save plot
    pylab.figure()
    pylab.plot(np.arange(no_epochs), hist)
    pylab.xlabel('epochs')
    pylab.ylabel(type_)
    pylab.title(name)
    if use_dropout:
        pylab.savefig(
            f'./results/partb1_{name}_{type_}_dropout.pdf')
    else:
        pylab.savefig(
            f'./results/partb1_{name}_{type_}_no_dropout.pdf')


def save_history(hist, name, use_dropout):
    # Create folder to store models and results
    if not os.path.exists('./results'):
        os.mkdir('./results')

    # save history
    if not os.path.exists('./histories'):
        os.mkdir('./histories')
    if use_dropout:
        hist_path = f'./histories/partb1_{name}_dropout'
    else:
        hist_path = f'./histories/partb1_{name}_no_dropout'
    with open(hist_path, 'wb') as file_pi:
        pickle.dump(hist, file_pi)


def save_model(model, name, use_dropout):
    if not os.path.exists('./models'):
        os.mkdir('./models')
    # Save model
    if use_dropout:
        model.save(f'./models/partb1_{name}_dropout')
    else:
        model.save(f'./models/partb1_{name}_no_dropout')


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_data_chars()
    # Use `tf.data` to batch and shuffle the dataset:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)

    # Build model
    tf.keras.backend.set_floatx('float32')
    model = CharCNN(256)

    # Choose optimizer and loss function for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    # Select metrics to measure the loss and the accuracy of the model.
    # These metrics accumulate the values over epochs and then print the overall result.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    test_acc = []
    train_acc = []
    test_cost = []
    train_cost = []
    for epoch in range(no_epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(model, images, labels, drop_rate=0)

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

    history = dict()
    history['train_accuracy'] = train_acc
    history['test_accuracy'] = test_acc
    history['test_loss'] = test_cost
    history['train_loss'] = train_cost
    save_model(model, 'Char_CNN', use_dropout=False)
    save_history(history, 'Char_CNN', use_dropout=False)
    save_plot(history['train_loss'], 'Char_CNN',
              'train_cost', use_dropout=False)
    save_plot(history['test_accuracy'], 'Char_CNN',
              'test_acc', use_dropout=False)
