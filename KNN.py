# Notwendige Bibliotheken importieren
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
#hier decomposition und mcnemar importieren

# CIFAR-10-Datenset laden
cifar10 = fetch_openml('CIFAR_10', version=1)

# Daten und Labels extrahieren
X = cifar10.data
y = cifar10.target.astype(np.int64)  # Labels müssen in Integer umgewandelt werden

# Daten vorverarbeiten
# Die Bilder haben die Dimensionen (32, 32, 3), also flachen wir sie zu Vektoren der Länge 3072 ab
X = X.reshape((X.shape[0], -1))

# Normalisierung der Daten
X= X.astype('float32') / 255.0

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Hier kommt Merkmalsextraktion hin

# Training des KNN-Modells
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train.ravel())

# Vorhersagen auf den Testdaten
y_pred = knn.predict(x_test)

# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit: {accuracy*100:.2f}%")

# KNN mit Merkmalextrahierer A
knn_A = KNeighborsClassifier(n_neighbors=3)
knn_A.fit(x_train_A, y_train.ravel())
y_pred_A = knn_A.predict(x_test_A)
accuracy_A = accuracy_score(y_test, y_pred_A)
print(f"Genauigkeit mit PCA: {accuracy_A*100:.2f}%")

