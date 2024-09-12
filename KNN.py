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


class KNNClassifier:
    def __init__(self, k=3, visu=False):
        self.k = k
        self.visu = visu
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            n_jobs=-1  # Use all available cores
        )

    def train(self, features, labels):
        if self.visu:
            print("Start Training Model...With Image Numbers:", len(features))
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def score(self, features, labels):
        if self.visu:
            print("Start Testing Model...With Image Numbers:", len(features))
        return self.model.score(features, labels)



def main(KNN_PCA=True, KNN_HOG=True):
    # Load Labels
    testLabel = np.load("dataset/dataset_split/testLabel.npy")
    trainLabel = np.load("dataset/dataset_split/trainLabel.npy")

    # PCA-KNN Classifer
    if KNN_PCA:
        pca_test = np.load("dataset/PCA/testPCA.npy")               # Load PCA features
        pca_train = np.load("dataset/PCA/trainPCA.npy")

        pca_classifier = KNNClassifier(k=3, visu=True)              # Create and train the classifiers
        pca_classifier.train(pca_train, trainLabel)                 # Train the classifiers
        pca_preds = pca_classifier.predict(pca_test)                # Test the classifiers

        pca_accuracy = pca_classifier.score(pca_test, testLabel)    # Calculate the accuracy
        print(f"PCA Accuracy: {pca_accuracy}")

    # HOG-KNN Classifer
    if KNN_HOG:
        hog_test = np.load('dataset/HOG/testHOG.npy')               # Load HOG features
        hog_train = np.load('dataset/HOG/trainHOG.npy')

        hog_classifier = KNNClassifier(k=18, visu=True)             # Create and train the classifiers

        hog_classifier.train(hog_train, trainLabel)                 # Test the classifiers
        hog_preds = hog_classifier.predict(hog_test)                # Test the classifiers

        hog_accuracy = hog_classifier.score(hog_test, testLabel)    # Calculate the accuracy
        print(f"HOG Accuracy: {hog_accuracy}")

    if KNN_PCA and KNN_HOG:
        # Compare the classifiers using p-value
        t_stat, p_value = stats.ttest_ind(pca_preds, hog_preds)
        print(f"P-value: {p_value}")

# set name if run directly
if __name__ == "__main__":
    main()