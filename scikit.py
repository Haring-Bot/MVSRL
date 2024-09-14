import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from skimage import color
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import readData


#Settings
PCAdimensions = 26
KNNneighbors = 25


class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, image_shape=(32, 32, 3), pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
        self.image_shape = image_shape
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reshape and apply HOG to each image in the dataset
        return [hog(color.rgb2gray(img.reshape(self.image_shape)), 
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    orientations=self.orientations) for img in X]


def main():
    #prepare data
    readData.main()
    
    trainData = np.load("dataset/dataset_split/trainData.npy")
    testData = np.load("dataset/dataset_split/testData.npy")
    trainLabel = np.load("dataset/dataset_split/trainLabel.npy")
    testLabel = np.load("dataset/dataset_split/testLabel.npy")

    bestAccuracy = 0 
    bestPCAdim = 0
    bestKNNneighbors = 0

    #for KNNneighbors in range(1, 100):
    #create PCA pipeline
    pipelinePCA = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components = PCAdimensions)),
        ("knn", KNeighborsClassifier(n_neighbors = KNNneighbors))
    ])

    #create HOG pipeline
    pipelineHOG = Pipeline([
        ("hog", HOGTransformer()),
        ("knn", KNeighborsClassifier(n_neighbors = KNNneighbors)) 
    ])

    #train pipelines
    pipelinePCA.fit(trainData, trainLabel)
    pipelineHOG.fit(trainData, trainLabel)

    #predict
    testPredictionsPCA = pipelinePCA.predict(testData)
    testPredictionsHOG = pipelineHOG.predict(testData)

    #evaluate
    accuracyPCA = accuracy_score(testLabel, testPredictionsPCA)
    accuracyHOG = accuracy_score(testLabel, testPredictionsHOG)
    print(f"Accuracy PCA: {accuracyPCA:.2f}")
    print(f"Accuracy HOG: {accuracyHOG:.2f}")


    if accuracyPCA > bestAccuracy:
        print(" ! new highest accuracy !")
        bestAccuracy = accuracyPCA
        bestPCAdim = PCAdimensions
        bestKNNneighbors = KNNneighbors

    print(f"PCA = {PCAdimensions:d}, KNNneighbors = {KNNneighbors:d}  Accuracy = {accuracyPCA:.2f}")
    print(f"The highest Accuracy of {bestAccuracy:.2f} was achieved with reducing dimensions via PCA to {bestPCAdim:d} and a KNN neighbors value of {bestKNNneighbors:d}")


if __name__ == "__main__":
    main()
