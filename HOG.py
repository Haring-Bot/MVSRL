# Lorant Albert
# 17/05/2024
# pca feature extraction

import numpy as np
from skimage.feature import hog
import pickle
# new
from skimage import color
import matplotlib.pyplot as plt

# function that can be used in the main code for extracting hog features
def compute_hog_features(data, visualize=False):
    all_features = []  # List to store features for all images
    # Normalize pixel values
    images_normalized = data.astype('float32') / 255.0

    setVisu = True
    for image in images_normalized:
        gray_image = color.rgb2gray(image)
        features, hog_image = hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True,
            feature_vector=True,
            transform_sqrt=True
        )

        if setVisu:
            image_visu = image
            hog_image_visu = hog_image
            setVisu = False
        all_features.append(features)  # Add features for this image to the list

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax1.imshow(image_visu)
        ax1.set_title('Original Image')
        ax2.imshow(hog_image_visu, cmap=plt.cm.gray)
        ax2.set_title('HOG Features')
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        #plt.show()

    return all_features

# Function to unpickle the dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

# code for testing code, not further important for main project
def main():
    # Load CIFAR dataset
    train_batch = np.load("dataset/dataset_split/trainData.npy")
    test_batch = np.load("dataset/dataset_split/testData.npy")
    # Reshape Image
    train_data = np.array([reshape_cifar_image(image) for image in train_batch])
    test_data = np.array([reshape_cifar_image(image) for image in test_batch])
    # Test show one image
    # plt.imshow(train_data[0])
    # plt.show()

    # Compute HOG features for training and testing data
    X_train_hog = compute_hog_features(train_data, visualize=True)
    X_test_hog = compute_hog_features(test_data, visualize=False)

    # Save HOG features to a file
    np.save('dataset/HOG/trainHOG.npy', X_train_hog)
    np.save('dataset/HOG/testHOG.npy', X_test_hog)
    print("Data saved successfully")

# set name if run directly
if __name__ == "__main__":
    main()
