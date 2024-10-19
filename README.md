## Iris Flowers Classification with PyTorch

## Overview
This project implements a multi-class classification problem using the Iris dataset and PyTorch. The model is designed to classify iris flowers into three species: **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica** based on four features (sepal length, sepal width, petal length, and petal width).

## Dataset
The dataset contains 150 samples, each with four features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Target labels:
- 0: Iris-setosa
- 1: Iris-versicolor
- 2: Iris-virginica

The dataset is split into training (80%) and testing (20%) sets.

## Project Structure
- **data/iris.data.csv**: The dataset used for training and testing.
- **train_model.py**: Main code for training the PyTorch model.
- **README.md**: Project documentation.
- **requirements.txt**: List of dependencies.

## Model Architecture
The model is a simple fully connected neural network with:
- Input layer: 4 features (sepal length, sepal width, petal length, petal width)
- Two hidden layers:
  - First hidden layer: 25 neurons
  - Second hidden layer: 30 neurons
- Output layer: 3 output classes (Iris species)
- Activation function: ReLU

## Training
The model is trained for 100 epochs using:
- **Loss function**: Cross-Entropy Loss
- **Optimizer**: Adam optimizer with a learning rate of 0.01

## Results
The model achieves an accuracy of around 96.6% on the test set.
