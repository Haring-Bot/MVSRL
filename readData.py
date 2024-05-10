# Lorant Albert
# data read

import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_cifar10_data(path_to_dataset):
    # Unpickle function
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    tar = tarfile.open(path_to_dataset, "r:gz")
    tar.extractall()
    tar.close()

    train_data = []
    train_labels = []

    # The CIFAR-10 dataset consists of 5 batches, named data_batch_1, data_batch_2, etc..
    for i in range(1, 6):
        data_dict = unpickle("cifar-10-batches-py/data_batch_" + str(i))
        if i == 1:
            train_data = data_dict[b'data']
            train_labels = data_dict[b'labels']
        else:
            train_data = np.vstack((train_data, data_dict[b'data']))
            train_labels = np.hstack((train_labels, data_dict[b'labels']))

    test_data_dict = unpickle("cifar-10-batches-py/test_batch")
    test_data = test_data_dict[b'data']
    test_labels = np.array(test_data_dict[b'labels'])

    return train_data, train_labels, test_data, test_labels


def main():
    path_to_dataset = "cifar-10-python.tar.gz"  # replace with your path
    train_data, train_labels, test_data, test_labels = load_cifar10_data(path_to_dataset)

    # Display the first image
    img = train_data[0].reshape((3, 32, 32)).transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()


# set name if run directly
if __name__ == "__main__":
    main()
