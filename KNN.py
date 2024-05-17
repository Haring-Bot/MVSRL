# Notwendige Bibliotheken importieren
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10

# CIFAR-10-Datenset laden
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Daten vorverarbeiten
# Die Bilder haben die Dimensionen (32, 32, 3), also flachen wir sie zu Vektoren der Länge 3072 ab
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Normalisierung der Daten
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#Hier kommt Merkmalsextraktion hin

# Training des KNN-Modells
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train.ravel())

# Vorhersagen auf den Testdaten
y_pred = knn.predict(x_test)

# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit: {accuracy*100:.2f}%")

# KNN mit PCA
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_pca.fit(x_train_pca, y_train.ravel())
y_pred_pca = knn_pca.predict(x_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"Genauigkeit mit PCA: {accuracy_pca*100:.2f}%")

