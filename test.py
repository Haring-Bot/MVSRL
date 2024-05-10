import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from knn_classifier import train_knn, evaluate_knn
from feature_extraction import apply_pca, apply_tsne
from tensorflow.keras.datasets import cifar10
from statsmodels.stats.contingency_tables import mcnemar

def mcnemars_test(y_true, pred1, pred2):
    """
    Perform McNemar's test on two sets of predictions.
    """
    table = np.zeros((2, 2), dtype=int)
    for a, b in zip(pred1, pred2):
        table[int(a)][int(b)] += 1
    result = mcnemar(table, exact=True)
    return result

def load_and_preprocess_cifar10():
    """
    Load and preprocess the CIFAR-10 dataset.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_train = x_train[:10000].reshape(10000, -1) / 255.0
    y_train = y_train[:10000].ravel()
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_test = y_test.ravel()
    return x_train, y_train, x_test, y_test

def main():
    """
    Main function to run KNN classification with PCA and t-SNE.
    """
    # Load CIFAR-10 data
   
