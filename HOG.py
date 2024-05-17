import numpy as np
from skimage.feature import hog
import pickle

NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 2000

def compute_hog_features(data):
    hog_features = []
    for image in data:
        # Reshape the image
        image = image.reshape(32, 32, 3)

        # Compute HOG features
        features = hog(
                        image,
                        orientations=8,
                        pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1),
                        visualize=False,
                        channel_axis=-1,
                       )

        hog_features.append(features)

    return np.array(hog_features)

# Function to unpickle the dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Load CIFAR dataset
train_batch = unpickle('cifar-10-batches-py/data_batch_1')
test_batch = unpickle('cifar-10-batches-py/test_batch')

# Extract the data
train_batch = train_batch[b'data'][:NUM_TRAIN_DATA]
test_batch = test_batch[b'data'][:NUM_TEST_DATA]

# Compute HOG features for training and testing data
X_train_hog = compute_hog_features(train_batch)
X_test_hog = compute_hog_features(test_batch)

# Save HOG features to a file
np.save('X_train_hog.npy', X_train_hog)
np.save('X_test_hog.npy', X_test_hog)

print("HOG features have been successfully saved to 'X_train_hog.npy' and 'X_test_hog.npy'")
