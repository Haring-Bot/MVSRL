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
from sklearn.model_selection import GridSearchCV

import readData


#Settings
PCAdimensions = 25
KNNneighbors = 16

PCAgrid = {
    "pca__n_components": range(5, 105, 10),
    "knn__n_neighbors": range(1, 30, 3)
}

HOGgrid = {
    'hog__pixels_per_cell': [(6, 6), (8, 8)],
    'hog__cells_per_block': [(2, 2), (3, 3)],
    'hog__orientations': [8, 9],
    'knn__n_neighbors': [5, 10, 25, 50]
    }


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
        ("pca", PCA()),
        ("knn", KNeighborsClassifier())
    ])

    #create HOG pipeline
    pipelineHOG = Pipeline([
        ("hog", HOGTransformer()),
        ("knn", KNeighborsClassifier(n_neighbors = KNNneighbors)) 
    ])

    #GridSearch for parameter tuning
    gridSearchPCA = GridSearchCV(pipelinePCA, PCAgrid, cv = 5, scoring = "accuracy")
    gridSearchHOG = GridSearchCV(pipelineHOG, HOGgrid, cv=5, scoring = "accuracy")

    #train pipelines
    print("start training...")
    gridSearchPCA.fit(trainData, trainLabel)
    gridSearchHOG.fit(trainData, trainLabel)

    #find best model
    bestModelPCA = gridSearchPCA.best_estimator_
    bestModelHOG = gridSearchHOG.best_estimator_
    
    #predict
    testPredictionsPCA = bestModelPCA.predict(testData)
    testPredictionsHOG = bestModelHOG.predict(testData)
    
    #accuracy
    accuracyPCA = accuracy_score(testLabel, testPredictionsPCA)
    accuracyHOG = accuracy_score(testLabel, testPredictionsHOG)
    
    #results
    print(f"\n Training completed! \nAn accuracy of {accuracyPCA:.2f} can be achieved with ", gridSearchPCA.best_params_)
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


if __name__ == "__main__":
    main()
