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


NUM_IMAGE_TRAIN = 10000
NUM_IMAGE_TEST = 20000
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



def main():

    # Load PCA and HOG features
    testPCA = np.load("dataset/PCA/testPCA.npy")
    trainPCA = np.load("dataset/PCA/trainPCA.npy")

    testHOG = np.load('dataset/HOG/testHOG.npy')
    trainHOG = np.load('dataset/HOG/trainHOG.npy')

    testLabel =np.load("dataset/dataset_split/testLabel.npy")
    trainLabel = np.load("dataset/dataset_split/trainLabel.npy")
    

    # Create and train the classifiers
    pca_classifier = KNNClassifier(k=3)
    pca_classifier.train(trainPCA, trainLabel)

    hog_classifier = KNNClassifier(k=3)
    hog_classifier.train(trainHOG, trainLabel)

    # Test the classifiers
    pca_preds = pca_classifier.predict(testPCA)
    hog_preds = hog_classifier.predict(testHOG)

    # Calculate the accuracy
    pca_accuracy = pca_classifier.score(testPCA, testLabel)
    hog_accuracy = hog_classifier.score(testHOG, testLabel)

    print(f"PCA Accuracy: {pca_accuracy}")
    print(f"HOG Accuracy: {hog_accuracy}")

    # Compare the classifiers using p-value
    t_stat, p_value = stats.ttest_ind(pca_preds, hog_preds)
    print(f"P-value: {p_value}")


# set name if run directly
if __name__ == "__main__":
    main()