from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import zipfile

# set up device for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading data
desired_files = []
overall_data = []
zip_path = "4543435.zip"

# extract force data files
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    desired_files = [name for name in sorted(zip_ref.namelist()) if name.endswith("forces.txt")]

# read each file and pad sequences
for file_name in desired_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref, zip_ref.open(file_name) as file:
        data = np.loadtxt(file, usecols=1, skiprows=1, dtype='float32')
        overall_data.append(data)

# padding data to make all sequences the same length
max_len = max(len(item) for item in overall_data)
overall_datanp = np.array([np.pad(item, (0, max_len - len(item))) for item in overall_data])

# load labels from excel
excel_filename = "39452935_RBDSinfo.xlsx"
desired_column = 'Injury'
desired_label_suffix = 'forces.txt'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    with zip_ref.open(excel_filename) as excel_file:
        df = pd.read_excel(excel_file)
    filtered_values = df.loc[df['FileName'].str.endswith(desired_label_suffix), desired_column].values

# convert labels to binary
filtered_values = np.where(filtered_values == "Yes", 1, 0).astype(np.float32)

# split the data into training and test sets
X_train, x_test, y_train, y_test = train_test_split(overall_datanp, filtered_values, test_size=0.25, random_state=42)

# converting data to torch tensors
X_train, y_train, x_test, y_test = map(torch.tensor, (X_train, y_train, x_test, y_test))

# create dataset objects
train = TensorDataset(X_train, y_train)
test = TensorDataset(x_test, y_test)

# set batch size and loaders
batch_size = 7
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# define model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3)  # convolutional layer
        self.fc1 = nn.Linear(16 * (max_len - 2), 1)  # fully connected layer

    def forward(self, x):
        x = x.view(-1, 1, max_len)  # reshape input
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # flatten for fc layer
        x = self.fc1(x)
        return x.squeeze()

# setting up model, loss, and optimizer
model = CNNModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# set number of epochs
num_epochs = 5

# setting up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# convert full dataset to TensorDataset
full_data = TensorDataset(torch.tensor(overall_datanp), torch.tensor(filtered_values))

# loop through each fold
for fold, (train_index, val_index) in enumerate(kf.split(full_data)):
    print(f'Fold [{fold + 1}]')

    # get train and validation subsets
    train_subset = torch.utils.data.Subset(full_data, train_index)
    val_subset = torch.utils.data.Subset(full_data, val_index)

    # set up loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)

    # reset model and optimizer
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_labels = []
        all_probabilities = []

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_labels.extend(y.cpu().detach().numpy())
            all_probabilities.extend(torch.sigmoid(outputs).cpu().detach().numpy())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / (i+1):.4f}')

    # validation loop
    model.eval()
    val_labels = []
    val_probabilities = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            val_labels.extend(y.cpu().detach().numpy())
            val_probabilities.extend(torch.sigmoid(outputs).cpu().detach().numpy())

    # calculate metrics
    val_predicted = (np.array(val_probabilities) > 0.5).astype(np.float32)
    val_accuracy = (val_predicted == np.array(val_labels)).sum().item() / len(val_labels)
    val_precision = precision_score(val_labels, val_predicted)
    val_recall = recall_score(val_labels, val_predicted)
    val_f1 = f1_score(val_labels, val_predicted)
    val_roc_auc = roc_auc_score(val_labels, val_probabilities)

    print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, ROC AUC: {val_roc_auc:.4f}')
