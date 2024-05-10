from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# CIFAR-10 Daten laden und vorbereiten
def load_and_preprocess_cifar10():
    from tensorflow.keras.datasets import cifar10
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

# KNN-Klassifikator trainieren und bewerten
def train_and_evaluate_knn(x_train, y_train, x_test, y_test, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    return accuracy, report

# Hauptfunktion
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_preprocess_cifar10()

    # KNN trainieren und bewerten
    accuracy, report = train_and_evaluate_knn(x_train, y_train, x_test, y_test, n_neighbors=3)

    print(f"KNN Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
