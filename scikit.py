import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from skimage import color
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import readData


#Settings

doPCA = True
doHOG = True

KNNneighbors = range(1, 51, 5)

#fixed values
PCAgrid = {
    "pca__n_components": range(1, 100, 10),
    "knn__n_neighbors": KNNneighbors
}

#optimise values
# PCAgrid = {
#     "pca__n_components": range(5, 105, 10),
#     "knn__n_neighbors": range(1, 30, 3)
# }

#fixed values
HOGgrid = {
    'hog__pixels_per_cell': [(4, 4)],
    'hog__cells_per_block': [(6, 6)],
    'hog__orientations': range(1, 50, 5),
    'knn__n_neighbors': KNNneighbors
    }

#optimise values
# HOGgrid = {
#     'hog__pixels_per_cell': [(4,4) ,(6,6), (8, 8)],
#     'hog__cells_per_block': [(3, 3), (6,6), (9,9)],
#     'hog__orientations': [5],
#     'knn__n_neighbors': range(1, 100, 10)
#     }


class ReshapeTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        #flatten images
        return X.reshape(X.shape[0], -1)


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
        return [hog(img,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    orientations=self.orientations,
                    channel_axis=2) for img in X]


def reshape_cifar_image(image):
    """
    Reshape a flattened CIFAR-10 image to its original format.
    """
    # Separate the color channels
    red = image[:1024].reshape(32, 32)
    green = image[1024:2048].reshape(32, 32)
    blue = image[2048:].reshape(32, 32)

    # Stack the color channels to create a 32x32x3 image
    return np.dstack((red, green, blue))


def main():
    
    #prepare data
    readData.main()

    trainData = np.load("dataset/dataset_split/trainData.npy")
    testData = np.load("dataset/dataset_split/testData.npy")
    trainLabel = np.load("dataset/dataset_split/trainLabel.npy")
    testLabel = np.load("dataset/dataset_split/testLabel.npy")

    trainData = np.array([reshape_cifar_image(image) for image in trainData])
    testData = np.array([reshape_cifar_image(image) for image in testData])

    # Test show one image
    #plt.imshow(trainData[0])
    #plt.show()

    print(f"trainData shape: {trainData.shape}")
    print(f"trainLabel shape: {trainLabel.shape}")
    print(f"testData shape: {testData.shape}")
    print(f"testLabel shape: {testLabel.shape}")

    bestAccuracy = 0 
    bestPCAdim = 0
    bestKNNneighbors = 0

    if doPCA:
        #create PCA pipeline
        pipelinePCA = Pipeline([
            ("reshape", ReshapeTransformer()),
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("knn", KNeighborsClassifier())
        ])
        #GridSearch for parameter tuning
        gridSearchPCA = GridSearchCV(pipelinePCA, PCAgrid, cv = 5, scoring = "accuracy")

        #train pipelines
        print("start training...")
        gridSearchPCA.fit(trainData, trainLabel)

        #find best model
        bestModelPCA = gridSearchPCA.best_estimator_

        #predict
        testPredictionsPCA = bestModelPCA.predict(testData)

        #accuracy
        accuracyPCA = accuracy_score(testLabel, testPredictionsPCA)

        #results
        print(f"\n Training completed! \nAn accuracy of {accuracyPCA:.2f} can be achieved with ", gridSearchPCA.best_params_)


    if doHOG:
        #create HOG pipeline
        pipelineHOG = Pipeline([
            ("hog", HOGTransformer()),
            ("knn", KNeighborsClassifier(n_neighbors = KNNneighbors)) 
        ])
        #GridSearch for parameter tuning
        gridSearchHOG = GridSearchCV(pipelineHOG, HOGgrid, cv=5, scoring = "accuracy")

        #train pipelines
        print("start training...")
        gridSearchHOG.fit(trainData, trainLabel)

        #find best model
        bestModelHOG = gridSearchHOG.best_estimator_
    
        #predict
        testPredictionsHOG = bestModelHOG.predict(testData)
    
        #accuracy
        accuracyHOG = accuracy_score(testLabel, testPredictionsHOG)
    
        #results
        print(f"\n Training completed! \nAn accuracy of {accuracyHOG:.2f} can be achieved with ", gridSearchHOG.best_params_)


    #evaluate
    #accuracyPCA = accuracy_score(testLabel, testPredictionsPCA)
    #accuracyHOG = accuracy_score(testLabel, testPredictionsHOG)
    #print(f"Accuracy PCA: {accuracyPCA:.2f}")
    #print(f"Accuracy HOG: {accuracyHOG:.2f}")


    # if accuracyPCA > bestAccuracy:
    #     print(" ! new highest accuracy !")
    #     bestAccuracy = accuracyPCA
    #     bestPCAdim = PCAdimensions
    #     bestKNNneighbors = KNNneighbors

    #print(f"PCA = {PCAdimensions:d}, KNNneighbors = {KNNneighbors:d}  Accuracy = {accuracyPCA:.2f}")
    #print(f"The highest Accuracy of {bestAccuracy:.2f} was achieved with reducing dimensions via PCA to {bestPCAdim:d} and a KNN neighbors value of {bestKNNneighbors:d}")

    print("End Code")

if __name__ == "__main__":
    main()
