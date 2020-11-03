import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.image as mpimg
import imageio as im
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model


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

# Fixed, no need change


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


def make_model(num_ch_c1, num_ch_c2, use_dropout):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(3072, )))
    model.add(layers.Reshape(target_shape=(32, 32, 3), input_shape=(3072,)))
    model.add(layers.Conv2D(filters=num_ch_c1, kernel_size=9,
                            activation='relu', padding='valid', input_shape=(None, None, 3)))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2), strides=2, padding="valid"))
    model.add(layers.Conv2D(filters=num_ch_c2, kernel_size=5,
                            activation='relu', padding='valid', input_shape=(None, None, 3)))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2), strides=2, padding="valid"))
    model.add(layers.Flatten())
    if use_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(300, activation=None, use_bias=True))
    if use_dropout:
        model.add(layers.Dropout(0.5))
    # Here no softmax because we have combined it with the loss
    model.add(layers.Dense(10, use_bias=True, input_shape=(300,)))
    return model


def save_model(model, use_dropout, num_ch_c1, num_ch_c2, optimizer_, history):
    # Create folder to store models and results
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./results'):
        os.mkdir('./results')

    # Save model
    if use_dropout:
        model.save(f'./models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout')
    else:
        model.save(f'./models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout')

    # save history
    if not os.path.exists('./histories'):
        os.mkdir('./histories')
    if use_dropout:
        hist_path = f'./histories/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout'
    else:
        hist_path = f'./histories/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout'
    with open(hist_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def save_plot(history, use_dropout, num_ch_c1, num_ch_c2, optimizer_):
    # Save the plot for losses
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Test')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    if use_dropout:
        plt.savefig(
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_loss.pdf')
    else:
        plt.savefig(
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_loss.pdf'
        )
    plt.close()

    # Save the plot for accuracies
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
    plt.plot(range(1, len(test_acc) + 1), test_acc, label='Test')
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    if use_dropout:
        plt.savefig(
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_accuracy.pdf'
        )
    else:
        plt.savefig(
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_accuracy.pdf'
        )
    plt.close()


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
        fig = plt.figure(figsize=(20, 8))
        for i in range(50):
            ax = fig.add_subplot(5, 10, i+1)
            channel_image = layer_activation[0, :, :, i]
            ax.matshow(channel_image, cmap='viridis')
        plt.title(layer_name)
        plt.grid(False)
        plt.savefig(f'./feature_maps/{name}_{layer_name}.png')
        # plt.show()


def main():
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    models = {}

    num_ch_c1 = 50
    num_ch_c2 = 100

    epochs = 1000  # Fixed
    batch_size = 128  # Fixed
    learning_rate = 0.001
    use_dropout = False
    optimizers_ = ['SGD-momentum', 'RMSProp', 'Adam']

    # Training and test
    x_train, y_train = load_data('./data/data_batch_1')
    x_test, y_test = load_data('./data/test_batch_trim')

    # -------------3 (a) - 3(c)-------------#
    # for optimizer_ in optimizers_:
    #     if optimizer_ == 'SGD':
    #         optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    #     elif optimizer_ == 'SGD-momentum':  # Question 3(a)
    #         optimizer = keras.optimizers.SGD(
    #             learning_rate=learning_rate, momentum=0.1)
    #     elif optimizer_ == 'RMSProp':  # Question 3(b)
    #         optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    #     elif optimizer_ == 'Adam':  # Question 3(c)
    #         optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    #     else:
    #         raise NotImplementedError(
    #             f'You do not need to handle [{optimizer_}] in this project.')

    #     model = make_model(num_ch_c1, num_ch_c2, use_dropout)
    #     loss = tf.keras.losses.SparseCategoricalCrossentropy(
    #         from_logits=True)
    #     # Training
    #     model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    #     history = model.fit(
    #         x_train,
    #         y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         validation_data=(x_test, y_test))
    #     # save model and the plots
    #     save_model(model, use_dropout, num_ch_c1,
    #                num_ch_c2, optimizer_, history)
    #     save_plot(history, use_dropout, num_ch_c1, num_ch_c2, optimizer_)

    # -------------3(d)-------------#
    use_dropout = True
    model = make_model(num_ch_c1, num_ch_c2, use_dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer_ = 'SGD'
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    # Training
    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test))
    # save model and the plots
    save_model(model, use_dropout, num_ch_c1,
               num_ch_c2, optimizer_, history)
    save_plot(history, use_dropout, num_ch_c1, num_ch_c2, optimizer_)


if __name__ == '__main__':
    main()
