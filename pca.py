from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt

NUM_DIMENSIONS = 100
NUM_TRAIN_DATA = 10000
NUM_TEST_DATA = 2000


def get_pca_features(image_data, dimensions, print_pca_data=False):
    #flatten data for pca
    #flat_image_data = image_data.reshape((len(image_data), -1))

    #compute pca
    pca = PCA(n_components=dimensions)
    fData = pca.fit_transform(image_data)

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
        reconstruction_error = np.mean((image_data - data_inverse) ** 2)
        print('Reconstruction error: ', reconstruction_error)

    return fData

def standardize(trainData, testData):
    scaler = StandardScaler()

    trainDataFlattend = trainData.reshape((len(trainData), -1))
    testDataFlattend = testData.reshape((len(testData), -1))

    trainDataStandardized = scaler.fit_transform(trainDataFlattend)
    testDataStandardized = scaler.transform(testDataFlattend)

    trainDataStandardized = trainDataStandardized.reshape(trainData.shape)
    testDataStandardized = testDataStandardized.reshape(testData.shape)

    return trainDataStandardized, testDataStandardized

def checkStandardization(data):
    # Flatten data if it's multidimensional
    flattened_data = data.reshape(-1, data.shape[-1])
    
    # Calculate mean and standard deviation
    mean = np.mean(flattened_data, axis=0)
    std_dev = np.std(flattened_data, axis=0)

    print(f"Mean of each feature: {mean}")
    print(f"Standard deviation of each feature: {std_dev}")

    # Check if mean is close to 0 and std deviation is close to 1
    standardized = np.allclose(mean, 0, atol=1e-2) and np.allclose(std_dev, 1, atol=1e-2)
    
    if standardized:
        print("Data is standardized.")
    else:
        print("Data is not standardized.")

def main():
    # Load CIFAR-10 data
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    train_data = np.load("dataset/dataset_split/trainData.npy")
    test_data = np.load("dataset/dataset_split/testData.npy")

    trainDataStandardized, testDataStandardized = standardize(train_data, test_data)

    checkStandardization(trainDataStandardized)

    # Apply PCA
    trainPCA = get_pca_features(trainDataStandardized, NUM_DIMENSIONS)
    testPCA = get_pca_features(testDataStandardized, NUM_DIMENSIONS)

    # Save the PCA transformed data
    np.save('dataset/PCA/testPCA.npy', testPCA)
    np.save('dataset/PCA/trainPCA.npy', trainPCA)

# set name if run directly
if __name__ == "__main__":
    main()



