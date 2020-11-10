import matplotlib.pyplot as plt
import pickle
import numpy as np


name_ = ['Char_RNN', 'Word_RNN']
improve_type_ = ['Vanilla', 'LSTM', '2Layers', 'GradClipping']
use_dropout = False


def plot_b6():
    '''
    plot training and testing acc for question B.6
    '''
    for name in name_:
        for improve_type in improve_type_:
            if use_dropout:
                hist_path = f'./histories/partb1_{name}_{improve_type}_dropout'
            else:
                hist_path = f'./histories/partb1_{name}_{improve_type}_no_dropout'
            with open(hist_path, 'rb') as file:
                history = pickle.load(file)
                plt.plot(range(1, len(history['train_accuracy'])+1),
                         history['accuracy'], label=f'training acc')
                plt.plot(range(1, len(history['test_accuracy'])+1),
                         history['test_accuracy'], label=f'testing acc')

    plt.title('Training and Testing Acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend()
    # plt.savefig(f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_acc.pdf')
    plt.show()


def print_b6_acc():
    '''
    print all optimisation acc for question B.6
    '''
    accs = dict()
    for name in name_:
        for improve_type in improve_type_:
            if use_dropout:
                hist_path = f'./histories/partb1_{name}_{improve_type}_dropout'
            else:
                hist_path = f'./histories/partb1_{name}_{improve_type}_no_dropout'
            with open(hist_path, 'rb') as file:
                history = pickle.load(file)
                accs[hist_path] = max(
                    history['test_accuracy'])

    sorted_accs = sorted(accs.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_accs:
        print(k, v)


def print_b1_acc():
    '''
    print accs extracted from histories: {CNN, RNN}X{Word, Char}
    '''
    accs = dict()
    for name in name_:
        if use_dropout:
            hist_path = f'./histories/partb1_{name}_dropout'
        else:
            hist_path = f'./histories/partb1_{name}_no_dropout'
        with open(hist_path, 'rb') as file:
            history = pickle.load(file)
            accs[hist_path] = max(
                history['test_accuracy'])

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


print_b6_acc()
