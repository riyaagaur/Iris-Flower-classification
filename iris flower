import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
dataset = pd.read_csv("../input/iris-dataset/iris.data.csv")
dataset.columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "species"]

# Map species to numerical values
mappings = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
dataset["species"] = dataset["species"].apply(lambda x: mappings[x])

# Split dataset into features and labels
X = dataset.drop("species", axis=1).values
y = dataset["species"].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert data into PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define the model
class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)                  
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)                  
        self.out = nn.Linear(hidden_layer2, output_features)      

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Instantiate the model
model = Model()

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# Plot training loss
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Evaluate the model
with torch.no_grad():
    y_pred_test = model(X_test)
    preds = y_pred_test.argmax(dim=1)
    accuracy = (preds == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Test with new data
unknown_iris = torch.tensor([4.0, 3.3, 1.7, 0.5])
with torch.no_grad():
    result = model(unknown_iris)
    print(f'Predicted class: {result.argmax().item()}')
