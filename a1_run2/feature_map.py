import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.image as mpimg
import imageio as im
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import Model


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32) / 255
    labels = np.array(labels, dtype=np.int32)
    return data, labels


def plot_featuremap(model, image, name):
    ''' Fill in Question 1(b) here. This website may help:
            https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
    '''

    layer_outputs = [layer.output for layer in model.layers[1:5]]
    # Creates a model that will return these outputs, given the model input
    activation_model = Model(
        inputs=model.input, outputs=layer_outputs)
    # Returns a list of five Numpy arrays: one array per layer activation
    activations = activation_model.predict(image)

    layer_names = []
    for layer in model.layers[1:5]:
        # Names of the layers, so you can have them as part of your plot
        layer_names.append(layer.name)

    images_per_row = 10
    # Displays the feature maps
    for layer_name, layer_activation in zip(layer_names, activations):

        # # Number of features in the feature map
        # n_features = layer_activation.shape[-1]
        # # The feature map has shape (1, size, size, n_features).
        # size = layer_activation.shape[1]
        # # Tiles the activation channels in this matrix
        # n_cols = n_features // images_per_row
        # display_grid = np.zeros((size * n_cols, images_per_row * size))
        # for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        #     for row in range(images_per_row):
        #         channel_image = layer_activation[0,
        #                                          :, :,
        #                                          col * images_per_row + row]
        #         # # Post-processes the feature to make it visually palatable
        #         # channel_image -= channel_image.mean()
        #         # channel_image /= channel_image.std()
        #         # channel_image *= 64
        #         # channel_image += 128
        #         # channel_image = np.clip(
        #         #     channel_image, 0, 255).astype('uint8')
        #         display_grid[col * size: (col + 1) * size,  # Displays the grid
        #                      row * size: (row + 1) * size] = channel_image
        # scale = 1. / size
        # plt.figure(figsize=(scale * display_grid.shape[1],
        #                     scale * display_grid.shape[0]))
        # plt.title(layer_name)
        # plt.grid(False)
        # plt.imshow(display_grid, aspect='auto', cmap='viridis')

        fig = plt.figure(figsize=(20, 8))

        for i in range(50):
            ax = fig.add_subplot(5, 10, i+1)
            channel_image = layer_activation[0, :, :, i]
            ax.matshow(channel_image, cmap='gray')
        plt.title(layer_name)
        plt.grid(False)
        plt.savefig(f'./a1_run2/feature_maps/{name}_{layer_name}.png')
        # plt.show()


def summary(model):
    # summarize feature map shapes
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name and 'pooling' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)


# ----------- main -------------#
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
models = {}

num_ch_c1 = 50
num_ch_c2 = 60

epochs = 1000  # Fixed
batch_size = 128  # Fixed
learning_rate = 0.001
use_dropout = False
optimizer_ = 'sgd'

# testing data
x_test, y_test = load_data('./a1_run2/data/test_batch_trim')

# load model
if use_dropout:
    model = keras.models.load_model(
        f'./a1_run2/models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout')
else:
    model = keras.models.load_model(
        f'./a1_run2/models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout')

plot_featuremap(model, x_test[0].reshape(1, 3072), 'test_image1')
plot_featuremap(model, x_test[1].reshape(1, 3072), 'test_image2')
