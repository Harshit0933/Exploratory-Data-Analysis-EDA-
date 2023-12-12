
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
# Load and preprocess the data
pf = pd.read_csv('Housing.csv')
pf['mainroad'].fillna(pf['mainroad'].mode()[0], inplace=True)
pf['guestroom'].fillna(pf['guestroom'].mode()[0], inplace=True)
# Check for NaN values
print(pf.isnull().sum())

# Handle NaN values
pf.dropna(inplace=True)

le = LabelEncoder()
pf['mainroad'] = le.fit_transform(pf['mainroad'])

scaler = StandardScaler()
pf_scaled = pd.DataFrame(scaler.fit_transform(pf[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]), columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking'])
pf_scaled['mainroad'] = pf['mainroad']

# Split the dataset
X_cls_nn = pf_scaled[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y_cls_nn = pf_scaled['mainroad']


X_train_cls_nn, X_test_cls_nn, y_train_cls_nn, y_test_cls_nn = train_test_split(X_cls_nn, y_cls_nn, test_size=0.25, random_state=42)

# Convert data to PyTorch tensors
X_train_cls_nn = torch.FloatTensor(X_train_cls_nn.values)
y_train_cls_nn = torch.FloatTensor(y_train_cls_nn.values).view(-1, 1)
X_test_cls_nn = torch.FloatTensor(X_test_cls_nn.values)
y_test_cls_nn = torch.FloatTensor(y_test_cls_nn.values).view(-1, 1)

# Define the neural network architecture for classification
class NeuralNetworkClassifier(nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model, loss function, and optimizer
model_cls_nn = NeuralNetworkClassifier()
criterion_cls = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer_cls = optim.Adam(model_cls_nn.parameters(), lr=0.001)

# Training the model
epochs_cls = 50
for epoch in range(epochs_cls):
    model_cls_nn.train()
    optimizer_cls.zero_grad()
    outputs_cls = model_cls_nn(X_train_cls_nn)
    loss_cls = criterion_cls(outputs_cls, y_train_cls_nn)
    loss_cls.backward()
    optimizer_cls.step()

# Evaluate the model
model_cls_nn.eval()
with torch.no_grad():
    y_pred_cls_nn = model_cls_nn(X_test_cls_nn)
    y_pred_cls_nn = (y_pred_cls_nn > 0.5).float()  # Convert probabilities to binary predictions

accuracy_cls_nn = accuracy_score(y_test_cls_nn.numpy(), y_pred_cls_nn.numpy())
print(f'Accuracy (Neural Network Classification): {accuracy_cls_nn:.2f}')
# Convert PyTorch tensors to NumPy arrays for plotting
y_test_cls_np = y_test_cls_nn.numpy().flatten()
y_pred_cls_np = y_pred_cls_nn.detach().numpy().flatten()

# Scatter plot for classification
plt.scatter(X_test_cls_nn[:, 0], X_test_cls_nn[:, 1], c=y_test_cls_np, cmap='coolwarm', marker='o', label='Actual Classes')
plt.scatter(X_test_cls_nn[:, 0], X_test_cls_nn[:, 1], c=y_pred_cls_np, cmap='coolwarm', marker='x', label='Predicted Classes')
plt.title('Classification: Actual vs. Predicted Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()