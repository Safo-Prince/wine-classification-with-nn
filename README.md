# Wine Classification using TensorFlow and Keras

This code demonstrates a simple machine learning model using TensorFlow and Keras to classify wines as either red or white based on their properties. The dataset used in this code consists of two separate datasets for red and white wines, which are combined to create a single dataset for training the model.

## Dependencies

Make sure you have the following libraries installed:

- NumPy (`import numpy as np`)
- Pandas (`import pandas as pd`)
- Matplotlib (`import matplotlib.pyplot as plt`)
- TensorFlow (`import tensorflow as tf`)
- scikit-learn (`from sklearn.model_selection import train_test_split`)

## Data

The code retrieves the red and white wine datasets from the UCI Machine Learning Repository using their respective URLs. The datasets are loaded into pandas DataFrames named `red` and `white`.

## Data Exploration

Some exploratory analysis is performed on the datasets, such as generating histograms of the alcohol content for red and white wines using Matplotlib. Additionally, the code adds a new column called 'type' to identify the wine type (red or white).

## Data Preparation

The red and white wine datasets are combined into a single dataset named `wines` using the `append` function in pandas. The 'type' column is added to distinguish between red and white wines. Then, the input features (`X`) are extracted from the dataset, and the target variable (`y`) is created by flattening the 'type' column.

## Data Splitting

The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn. The training set is used to train the model, while the testing set is used to evaluate its performance. The split is performed with a test size of 0.34 and a random state of 42 for reproducibility.

## Model Architecture

A sequential model is created using Keras, which consists of three layers: a flatten layer, followed by two dense layers with ReLU activation, and a final dense layer with a sigmoid activation. The model is compiled with binary cross-entropy loss, the Adam optimizer, and accuracy as the evaluation metric.

## Model Training

The model is trained using the `fit` function, with the training data (`X_train` and `y_train`) and validation data (`X_test` and `y_test`). The training is performed over 8 epochs with a batch size of 1. The training history is stored, including accuracy and loss values for both the training and validation sets.

## Model Evaluation

The training history is used to plot the training and validation accuracy, as well as the training and validation loss over the epochs. Two subplots are created using Matplotlib to visualize the results.

## Prediction

Finally, a sample test data point is created and reshaped to match the input shape of the model. The model's `predict` function is used to make a prediction on the test data, returning the probability of the wine being either red or white.

Please note that this code assumes a basic understanding of machine learning concepts, Python programming, and the mentioned libraries. Feel free to modify the code or experiment with different parameters to further explore wine classification using this model.
