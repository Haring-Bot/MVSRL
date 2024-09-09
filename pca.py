from sklearn.decomposition import PCA
import numpy as np
import pickle
import matplotlib.pyplot as plt

NUM_DIMENSIONS = 100
NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 2000


def get_pca_features(image_data, dimensions, print_pca_data=False):
    #flatten data for pca
    flat_image_data = image_data.reshape((len(image_data), -1))

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

    return fData

def main():
    # Load CIFAR-10 data
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    train_data = np.load("dataset/dataset_split/trainData.npy")
    test_data = np.load("dataset/dataset_split/testData.npy")

    # Apply PCA
    trainPCA = get_pca_features(train_data, NUM_DIMENSIONS)
    testPCA = get_pca_features(test_data, NUM_DIMENSIONS)

    # Save the PCA transformed data
    np.save('dataset/PCA/testPCA.npy', testPCA)
    np.save('dataset/PCA/trainPCA.npy', trainPCA)

# set name if run directly
if __name__ == "__main__":
    main()



