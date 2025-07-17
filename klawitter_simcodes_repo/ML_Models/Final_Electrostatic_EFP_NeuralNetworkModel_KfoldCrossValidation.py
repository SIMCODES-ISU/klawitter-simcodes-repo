import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\Klaws\Documents\sij_rij_coul_full_dataset.xlsx")
grouped = df.groupby("File")

X = []
y = []

for file_name, group in grouped:
    coul_energy = float(group["Coul "].iloc[0])
    rij_vector = group["RIJ"].tolist()
    sij_vector = group["SIJ"].tolist()
    ground_truth = float(group["SAPT_GT"].iloc[0])
    X.append([coul_energy, rij_vector, sij_vector])
    y.append(ground_truth)

max_rij_len = max(len(sample[1]) for sample in X)
max_sij_len = max(len(sample[2]) for sample in X)

def pad_list(lst, target_length):
    return lst + [0.0] * (target_length - len(lst)) if len(lst) < target_length else lst[:target_length]

X_flat = []
for sample in X:
    coul_energy = sample[0]
    rij_vector = pad_list(sample[1], max_rij_len)
    sij_vector = pad_list(sample[2], max_sij_len)
    X_flat.append([coul_energy] + rij_vector + sij_vector)

X = np.array(X_flat, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

# === Define Model ===
class ResidualBlock(nn.Module):
    def __init__(self, size, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.norm1 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(size, size)
        self.norm2 = nn.LayerNorm(size)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out += residual
        return self.activation(out)

class EnergyPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(512, 0.3), ResidualBlock(512, 0.3), ResidualBlock(512, 0.3)
        )
        self.downsample = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SELU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.SELU(), nn.Dropout(0.25),
        )
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.downsample(x)
        return self.output_layer(x)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
mae_scores, rmse_scores, r2_scores = [], [], []

for train_index, test_index in kf.split(X):
    print(f"Fold {fold}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    model = EnergyPredictor(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)
        y_pred = y_pred_t.numpy()
        y_true = y_test_t.numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}\n")
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    fold += 1

print("=== Cross-Validation Results ===")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
