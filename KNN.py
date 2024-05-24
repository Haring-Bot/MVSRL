from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tarfile
from urllib.request import urlretrieve
import os
from skimage.feature import hog
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats


NUM_IMAGE_TRAIN = 1000
NUM_IMAGE_TEST = 2000
NUM_DIMENSIONS = 100
NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 2000


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    def train(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def score(self, features, labels):
        return self.model.score(features, labels)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#function that can be used in main:
def get_pca_features(image_data, dimensions, print_pca_data=False):

    #normalize mean 0 std-deviation 1
    scaler = StandardScaler()
    image_data_scaled = scaler.fit_transform(image_data)

    #flatten data for pca --> 3 channel to 1 channel (1D)
    flat_image_data = image_data_scaled.reshape((len(image_data_scaled), -1))

    #compute pca
    pca = PCA(n_components=dimensions)
    fData = pca.fit_transform(flat_image_data)

    # DEBUGING --> only if Flag is on true
    if print_pca_data:
        # print variance per dimension
        print('Explained variance by each component: ', pca.explained_variance_ratio_)

        # plot variance over components
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.show()

        # print accuracy by reconstruct image from pca
        data_inverse = pca.inverse_transform(fData)
        reconstruction_error = np.mean((flat_image_data - data_inverse) ** 2)
        print('Reconstruction error: ', reconstruction_error)

    return fData #return pca data


# function that can be used in the main code for extracting hog features
def compute_hog_features(data):
    hog_features = []
    for image in data:
        # Reshape the image
        image = image.reshape(32, 32, 3)

        # Compute HOG features
        features = hog(
                        image,
                        orientations=8,
                        pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1),
                        visualize=False,
                        channel_axis=-1,
                       )

        hog_features.append(features)

    return np.array(hog_features)


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

    X_train_pca = get_pca_features(train_data, NUM_DIMENSIONS)
    X_test_pca = get_pca_features(test_data, NUM_DIMENSIONS)

    # Save the PCA transformed data
    np.save('dataset/pca/train_pca.npy', X_train_pca)
    np.save('dataset/pca/test_pca.npy', X_test_pca)

    # Compute HOG features for training and testing data
    X_train_hog = compute_hog_features(train_data)
    X_test_hog = compute_hog_features(test_data)

    # Save HOG features to a file
    np.save('dataset/hog/train_hog.npy', X_train_hog)
    np.save('dataset/hog/test_hog.npy', X_test_hog)

    print("HOG features have been successfully saved to 'X_train_hog.npy' and 'X_test_hog.npy'")



    # Load PCA and HOG features
    pca_train = np.load('dataset/pca/train_pca.npy')
    pca_test = np.load('dataset/pca/test_pca.npy')
    hog_train = np.load('dataset/hog/train_hog.npy')
    hog_test = np.load('dataset/hog/test_hog.npy')

    # Create and train the classifiers
    pca_classifier = KNNClassifier(k=3)
    pca_classifier.train(pca_train, train_labels)

    hog_classifier = KNNClassifier(k=3)
    hog_classifier.train(hog_train, train_labels)

    # Test the classifiers
    pca_preds = pca_classifier.predict(pca_test)
    hog_preds = hog_classifier.predict(hog_test)

    # Calculate the accuracy
    pca_accuracy = pca_classifier.score(pca_test, test_labels)
    hog_accuracy = hog_classifier.score(hog_test, test_labels)

    print(f"PCA Accuracy: {pca_accuracy}")
    print(f"HOG Accuracy: {hog_accuracy}")

    # Compare the classifiers using p-value
    t_stat, p_value = stats.ttest_ind(pca_preds, hog_preds)
    print(f"P-value: {p_value}")


# set name if run directly
if __name__ == "__main__":
    main()