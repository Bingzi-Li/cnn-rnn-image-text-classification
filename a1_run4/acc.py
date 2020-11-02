import matplotlib.pyplot as plt
import pickle


ch_c1 = [10, 30, 50, 70, 90]
ch_c2 = [20, 40, 60, 80, 100]
optimizer_ = 'SGD'


def plot(ch_c1, ch_c2, optimizer_):
    for num_ch_c1 in ch_c1:
        for num_ch_c2 in ch_c2:
            hist_path = f'./a1_run3//histories/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout'
            with open(hist_path, 'rb') as file:
                history = pickle.load(file)
                plt.plot(range(1, len(history['val_accuracy'])+1),
                         history['val_accuracy'], label=f'c1:{num_ch_c1}_c2:{num_ch_c2}')
    plt.title('Test Acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.show()


def print_acc(ch_c1, ch_c2, optimizer_):
    accs = dict()
    for num_ch_c1 in ch_c1:
        for num_ch_c2 in ch_c2:
            hist_path = f'./a1_run4//histories/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout'
            with open(hist_path, 'rb') as file:
                history = pickle.load(file)
                accs[f'c1:{num_ch_c1}_c2:{num_ch_c2}'] = max(
                    history['val_accuracy'])

    sorted_accs = sorted(accs.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_accs:
        print(k, v)


print_acc(ch_c1, ch_c2, optimizer_)
