import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# extract data from zip file
zip_path = "4543435.zip"
desired_files = []
overall_data = []

# get list of files ending with "forces.txt"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    desired_files = [name for name in sorted(zip_ref.namelist()) if name.endswith("forces.txt")]

# read each file and store data
for file_name in desired_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref, zip_ref.open(file_name) as file:
        data = np.loadtxt(file, usecols=1, skiprows=1, dtype='float32')
        overall_data.append(data)

# pad data so all sequences have the same length
max_len = max(len(item) for item in overall_data)
overall_datanp = np.array([np.pad(item, (0, max_len - len(item))) for item in overall_data])

# load labels from the excel file
excel_filename = "39452935_RBDSinfo.xlsx"
desired_column = 'Injury'
desired_label_suffix = 'forces.txt'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    with zip_ref.open(excel_filename) as excel_file:
        df = pd.read_excel(excel_file)
    filtered_values = df.loc[df['FileName'].str.endswith(desired_label_suffix), desired_column].values

# convert labels to boolean format
labels_np = np.array([True if binary == "Yes" else False for binary in filtered_values])

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(overall_datanp, labels_np, test_size=0.25, random_state=42)

# convert data to torch tensors
x_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# reshape tensors to add an extra dimension
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

# define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1])  # get the last time step
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# set model parameters
input_dim = x_train.shape[2]
hidden_dim = 128
output_dim = 2

# initialize the model, loss function, and optimizer
model = LSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # reset gradients

    # forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # backward pass and optimization
    loss.backward()
    optimizer.step()

    # print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    # convert to numpy for scikit-learn metrics
    predicted_np = predicted.numpy()
    y_test_np = y_test.numpy()

    # calculate additional metrics
    precision = precision_score(y_test_np, predicted_np)
    recall = recall_score(y_test_np, predicted_np)

    # calculate probabilities for ROC/AUC
    test_output_prob = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
    roc_auc = roc_auc_score(y_test_np, test_output_prob)

    print(f'Testing Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')

    # calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_np, test_output_prob)

# plot ROC curve
plt.figure()
plt.grid(True)
plt.plot(fpr, tpr, label=f'(AUC = {roc_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
