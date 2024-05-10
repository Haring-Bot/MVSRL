from sklearn.decomposition import PCA
import numpy as np
import pickle

def main():
    # Set the number of train and test data
    n_train = 10000  # Change this to your desired number of train data
    n_test = 2000    # Change this to your desired number of test data


    # Load CIFAR-10 data
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    # Assuming you have the CIFAR-10 data in the same directory and it's named 'cifar-10-batches-py'
    train_data = unpickle('cifar-10-batches-py/data_batch_1')
    test_data = unpickle('cifar-10-batches-py/test_batch')

    X_train = train_data[b'data'][:n_train]
    X_test = test_data[b'data'][:n_test]

    # Flatten the images for PCA
    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))

    # Apply PCA
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Save the PCA transformed data
    np.save('X_train_pca.npy', X_train_pca)
    np.save('X_test_pca.npy', X_test_pca)

    print("finised")

# set name if run directly
if __name__ == "__main__":
    main()



