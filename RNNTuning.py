import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import zipfile
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading data from zip file
desired_files = []
overall_data = []
zip_path = "4543435.zip"

# extracting force data files
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    desired_files = [name for name in sorted(zip_ref.namelist()) if name.endswith("forces.txt")]

# read and pad each file
for file_name in desired_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref, zip_ref.open(file_name) as file:
        data = np.loadtxt(file, usecols=1, skiprows=1, dtype='float32')
        overall_data.append(data)

# pad sequences to the longest length
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

# split data into training and test sets
X_train, x_test, y_train, y_test = train_test_split(overall_datanp, filtered_values, test_size=0.25, random_state=42)
X_train, y_train, x_test, y_test = map(torch.tensor, (X_train, y_train, x_test, y_test))

# create TensorDatasets and DataLoaders
train = TensorDataset(X_train, y_train)
test = TensorDataset(x_test, y_test)

# initial batch size for loaders
batch_size = 8
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# define RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)  # initial hidden state
        outputs, hn = self.rnn(x, h_0)
        outputs = self.fc(outputs[:, -1, :])  # take the last time step
        return outputs.squeeze()

input_dim = 10
output_dim = 1
sequence_dim = 900

# objective function for Optuna optimization
def objective(trial):
    # define hyperparameters to tune
    batch_size = trial.suggest_int("batch_size", 2, 8, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 1, 100)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    num_epochs = trial.suggest_int("num_epochs", 1, 50)

    # setup loaders with suggested batch size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    # initialize model and loss function
    model = RNNModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # select optimizer based on suggested type
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(-1, sequence_dim, input_dim).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # report intermediate loss to Optuna
        intermediate_value = epoch_loss / (batch_idx + 1)
        trial.report(intermediate_value, epoch)

        # check if trial should be pruned
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, sequence_dim, input_dim).to(device)
            y = y.to(device)
            test_outputs = model(x)
            predicted = (torch.sigmoid(test_outputs) > 0.5).float()
            total += y.size(0)
            correct += (predicted == y).sum().item()

    # calculate accuracy
    accuracy = correct / total
    return accuracy

# create Optuna study and run optimization
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

# print best results
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
