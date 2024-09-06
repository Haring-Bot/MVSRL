# Lorant Albert
# read data of CIFAR
# download: https://www.cs.toronto.edu/~kriz/cifar.html
# implementation in other file:
#   1. from readData import load_cifar10_data
#   2. path_to_dataset = "cifar-10-python.tar.gz"  # replace with your path
#   3. train_data, train_labels, test_data, test_labels = load_cifar10_data(path_to_dataset)

import tarfile
import pickle
import numpy as np
from urllib.request import urlretrieve
import os

NUM_IMAGE_TRAIN = 10000
NUM_IMAGE_TEST = 2000

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(path_to_dataset):
    # Unpickle function
    pathToBatches = "dataset/cifar-10-batches-py/"

    tar = tarfile.open(path_to_dataset, "r:gz")
    tar.extractall(path = "dataset/")
    tar.close()

    train_data = []
    train_labels = []

    print("trying to load the dataset")
    # The CIFAR-10 dataset consists of 5 batches, named data_batch_1, data_batch_2, etc..
    for i in range(1, 6):
        data_dict = unpickle(pathToBatches+ "data_batch_" + str(i))
        if i == 1:
            train_data = data_dict[b'data']
            train_labels = data_dict[b'labels']
        else:
            train_data = np.vstack((train_data, data_dict[b'data']))
            train_labels = np.hstack((train_labels, data_dict[b'labels']))

    test_data_dict = unpickle(pathToBatches + "test_batch")
    test_data = test_data_dict[b'data']
    test_labels = np.array(test_data_dict[b'labels'])

    # Select a subset for training
    train_data_subset = train_data[:NUM_IMAGE_TRAIN]
    train_labels_subset = train_labels[:NUM_IMAGE_TRAIN]

    # Select a subset for testing
    test_data_subset = test_data[:NUM_IMAGE_TEST]
    test_labels_subset = test_labels[:NUM_IMAGE_TEST]

    return train_data_subset, train_labels_subset, test_data_subset, test_labels_subset

def downloadDataset(url):
    filename = "downloads/CIFAR-10_pickled.tar.gz"

    if os.path.exists(filename):
        print("found data locally")
        return filename

    try:
        print("downloading data from:", url)
        urlretrieve(url, filename)
        print("download complete")
        return filename
    except Exception as e:
        print(f"An error occurred while downloading: {e}")
        return None  # Return None or raise the exception based on your needs




def main():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    path = downloadDataset(url)
    train_data, train_labels, test_data, test_labels = load_cifar10_data(path)

    # Display the first image
    img = train_data[0].reshape((3, 32, 32)).transpose((1, 2, 0))
    print("Train:" + str(len(train_data)) + " / Test: " + str(len(test_data)))


# set name if run directly
if __name__ == "__main__":
    main()
