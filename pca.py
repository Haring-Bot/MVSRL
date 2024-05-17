from sklearn.decomposition import PCA
import numpy as np
import pickle
import matplotlib.pyplot as plt

NUM_DIMENSIONS = 100
NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 2000


def get_pca_features(imageData, dimensions, print_pca_data=False):
    pca = PCA(n_components=dimensions)
    fData = pca.fit_transform(imageData)

    # DEBUGING
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
        reconstruction_error = np.mean((imageData - data_inverse) ** 2)
        print('Reconstruction error: ', reconstruction_error)

    return fData

def main():
    # Load CIFAR-10 data
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    # Assuming you have the CIFAR-10 data in the same directory and it's named 'cifar-10-batches-py'
    train_data = unpickle('cifar-10-batches-py/data_batch_1')
    test_data = unpickle('cifar-10-batches-py/test_batch')

    X_train = train_data[b'data'][:NUM_TRAIN_DATA]
    X_test = test_data[b'data'][:NUM_TEST_DATA]

    # Flatten the images for PCA
    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))

    # Apply PCA
    X_train_pca = get_pca_features(X_train, NUM_DIMENSIONS, False)
    X_test_pca = get_pca_features(X_test, NUM_DIMENSIONS)

    # Save the PCA transformed data
    np.save('X_train_pca.npy', X_train_pca)
    np.save('X_test_pca.npy', X_test_pca)

# set name if run directly
if __name__ == "__main__":
    main()



