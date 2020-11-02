import matplotlib.pyplot as plt
import pickle
import numpy as np


ch_c1 = [50]
ch_c2 = [100]
optimizer_ = 'SGD'


def plot(ch_c1, ch_c2, optimizer_):
    for num_ch_c1 in ch_c1:
        for num_ch_c2 in ch_c2:
            hist_path = f'./a1_run6//histories/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout'
            with open(hist_path, 'rb') as file:
                history = pickle.load(file)
                plt.plot(range(1, len(history['accuracy'])+1),
                         history['accuracy'], label=f'training acc')
                plt.plot(range(1, len(history['val_accuracy'])+1),
                         history['val_accuracy'], label=f'testing acc')

    plt.title('Training and Testing Acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(
        f'./a1_run6/results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_acc.pdf')
    plt.show()


def print_acc(ch_c1, ch_c2, optimizer_):
    accs = dict()
    for num_ch_c1 in ch_c1:
        for num_ch_c2 in ch_c2:
            hist_path = f'./a1_run6//histories/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout'
            with open(hist_path, 'rb') as file:
                history = pickle.load(file)
                accs[f'c1:{num_ch_c1}_c2:{num_ch_c2}'] = max(
                    history['val_accuracy'])

    sorted_accs = sorted(accs.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_accs:
        print(k, v)


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


print_acc(ch_c1, ch_c2, optimizer_)
# x_train, y_train = load_data('./a1_run6/data/data_batch_1')
# for y in y_train[:300]:
#     print(y)
