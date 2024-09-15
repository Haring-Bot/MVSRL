# PCA/KNN vs HOG/KNN
## Overview
For the *Moderne Verfahren zur sensorbasierten Roboterregelung* laboratory the difference between **PCA** and another feature extractor should be compared. As the classifier **KNN** should be used with the **CIFAR-10** dataset. In the end the results are compared using the **McNemar's test**.

## Requirements
Install the following python libraries before executing the main:
- numpy
- scikit-learn
- scikit-image
- matplotlib

## Dataset
When executing the main the CIFAR-10 dataset is downlaoded from the website of the university of Toronto.
The CIFAR-10 dataset contains 60.000 32x32 images split into 10 different classes.
The data is split into:
- TrainData
- TrainLabel
- TestData
- TestLabel

## Feature Extractors
### PCA (Principal Component Analysis)
PCA is used to reduce the dimensionality of the dataset. The pipeline standardizes the data and applies PCA to project it into a lower-dimensional space.
#### Pipeline
1. Reshapes images.
2. Standardizes the data using StandardScaler.
3. Applies PCA for dimensionality reduction.
4. Classifies using KNN.

### HOG
HOG extracts features from images by calculating the gradient orientations.
#### Pipeline
1. Applies HOG transformation to the images.
2. Classifies using KNN.

## Hyperparameter tuning
For attaining the optimal hyperparameters for both pipelines **Grid Search** is implemented. Grid Search trains and evaluates the model with a range of parameters to determien the optimal set. The range can be set using the **PCAgrid** and **HOGgrid** arrays.

## Workflow 
1. **Data preparation:** The CIFAR-10 dataset is downloaded and split into train and test sets as well as data and label sets.
2. **Training:** The PCA and HOG pipelines are trained and evaluated trying to find the optimal parameetrs using Grid Search.
3. **Evaluation:** The best models of both pipelines are evaluated and the accuracy printed.
4. **McNemar:** Both pipeliens are compared to each other using the McNemar test.

## Settings
The most important settings:
- **doPCA:** enables the training and evaluation of the PCA pipeline
- **doHOG:** enables the training and evaluation of the HOG pipeline
**!!** attention: McNemar's test is only calculated if both doPCA and doHOG are set True **!!**
- ** KNNneighbors: ** sets the amount of neighbors for the KNN algorithm. Is automatically set for both pipelines.

## run
To run this project simply install all dependencies and execute the main.
`python main.py`

## Authors
- [Julian Haring](https://github.com/Haring-Bot) - Creator
- [Lorant Albert](https://github.com/AlbertL98) - Creator
- [Franz Schwarz](https://github.com/re23m023) - Creator