from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Beispiel: CIFAR-10 Daten laden
def load_and_preprocess_cifar10():
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Einfaches Preprocessing (Normierung)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Daten reshapen und Labels konvertieren
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_and_preprocess_cifar10()
