import os
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
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix

import readData


#Settings

doPCA = True
doHOG = True

KNNneighbors = [20]

#fixed values
PCAgrid = {
    "pca__n_components": [25],
    "knn__n_neighbors": KNNneighbors
}

#optimise values
# PCAgrid = {
#     "pca__n_components": range(5, 105, 5),
#     "knn__n_neighbors": KNNneighbors
# }

#fixed values
HOGgrid = {
    'hog__pixels_per_cell': [(4, 4)],
    'hog__cells_per_block': [(6, 6)],
    'hog__orientations': [8],
    'knn__n_neighbors': KNNneighbors
    }

# #optimise values
# HOGgrid = {
#     'hog__pixels_per_cell': [(4,4) ,(6,6), (8, 8), (10,10)],
#     'hog__cells_per_block': [(3, 3), (6,6), (9,9), (12,12)],
#     'hog__orientations': range(1, 100, 5),
#     'knn__n_neighbors': KNNneighbors
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
    #readData.main()

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
        gridSearchPCA = GridSearchCV(pipelinePCA, PCAgrid, scoring = "accuracy")    #without cross validations
        #gridSearchPCA = GridSearchCV(pipelinePCA, PCAgrid, cv = 5, scoring = "accuracy")   #with cross validation

        #train pipelines
        print("\nstarted training PCA...")
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
        gridSearchHOG = GridSearchCV(pipelineHOG, HOGgrid, scoring = "accuracy")            #without cross validation
        #gridSearchHOG = GridSearchCV(pipelineHOG, HOGgrid, cv=5, scoring = "accuracy")      #with crossvalidation

        #train pipelines
        print("\nstarted training HOG...")
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

    if doPCA and doHOG:
        print("calculating McNemar's Test")
        contingencyTable = np.zeros((2,2), dtype = int)

        for nPCApred, nHOGpred, nLabel in zip(testPredictionsPCA, testPredictionsHOG, testLabel):
            if nPCApred == nLabel and nHOGpred == nLabel:   #both right
                contingencyTable[0, 0] += 1
            elif nPCApred == nLabel and nHOGpred != nLabel: #only PCA right
                contingencyTable[0, 1] += 1  
            elif nPCApred != nLabel and nHOGpred == nLabel: #only HOG right
                contingencyTable[1, 0] += 1 
    else:                                                   #both wrong
                contingencyTable[1, 1] += 1 

    print(f"\n Contingency Table:\n{contingencyTable}\n")

    mcNResult = mcnemar(contingencyTable, exact = True)

    alpha = 0.05
    print(mcNResult)
    print("alpha = ", alpha)

    # Interpret the p-value
    if mcNResult.pvalue < alpha:
        print("\nThe difference between the PCA and HOG models is statistically significant.")
    else:
        print("\nThe difference between the PCA and HOG models is not statistically significant.")

    #print(f"\nMcNemar's test:\n{mcNResult.statistic}, \n p-value: {mcNResult.pvalue}")
    #print("End Code")

if __name__ == "__main__":
    main()
