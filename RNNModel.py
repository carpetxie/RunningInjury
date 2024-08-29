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

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import optuna
# Load Data
desired_files = []
overall_data = []
zip_path = "4543435.zip"

# X
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    desired_files = [name for name in sorted(zip_ref.namelist()) if name.endswith("forces.txt")]

for file_name in desired_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref, zip_ref.open(file_name) as file:
        data = np.loadtxt(file, usecols=1, skiprows=1, dtype='float32')
        overall_data.append(data)

max_len = max(len(item) for item in overall_data)
overall_datanp = np.array([np.pad(item, (0, max_len - len(item))) for item in overall_data])  # Padding data for equal dimensions

# Labels
excel_filename = "39452935_RBDSinfo.xlsx"
desired_column = 'Injury'
desired_label_suffix = 'forces.txt'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    with zip_ref.open(excel_filename) as excel_file:
        df = pd.read_excel(excel_file)
    filtered_values = df.loc[df['FileName'].str.endswith(desired_label_suffix), desired_column].values

filtered_values = np.where(filtered_values == "Yes", 1, 0).astype(np.float32)

# Splitting data
X_train, x_test, y_train, y_test = train_test_split(overall_datanp, filtered_values, test_size=0.25, random_state=42)
X_train, y_train, x_test, y_test = map(torch.tensor, (X_train, y_train, x_test, y_test))
train = TensorDataset(X_train, y_train)
test = TensorDataset(x_test, y_test)

batch_size = 7
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_dim).to(device)
        outputs, hn = self.rnn(x, h_0)
        outputs = self.fc(outputs[:, -1, :])
        return outputs.squeeze()


input_dim = 10
hidden_dim = 59
output_dim = 1
sequence_dim = 900

model = RNNModel(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.03775545428799915)

# Training
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    all_labels = []
    all_probabilities = []

    for i, (x, y) in enumerate(train_loader):
        x = x.view(-1, sequence_dim, input_dim).to(device)
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

# Metrics calculation after training
predicted = (np.array(all_probabilities) > 0.5).astype(np.float32)
epoch_accuracy = (predicted == np.array(all_labels)).sum().item() / len(all_labels)
epoch_precision = precision_score(all_labels, predicted)
epoch_recall = recall_score(all_labels, predicted)
epoch_f1 = f1_score(all_labels, predicted)
epoch_roc_auc = roc_auc_score(all_labels, all_probabilities)

print(f'Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1 Score: {epoch_f1:.4f}, ROC AUC: {epoch_roc_auc:.4f}')

fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
plt.plot(fpr, tpr, label=f'AUC = {epoch_roc_auc:.4f}')
plt.legend(loc='best')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# Evaluation
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

print(f'Testing Accuracy: {correct/total:.4f}')
