
import pandas as pd
from model.loader import CustomDataset
from model.model import Model
from model.trainer import train_model
from torch.utils.data import DataLoader
from analysis.data_split import stratified_split
from analysis.logger import Logger


import torch
import torch.optim as optim
import torch.nn as nn
import os 



df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
sizes = [len(df.columns)-1, 32,16,1]
output_activation = nn.Sigmoid
intermediate_activation = nn.ReLU
normalize_output = True
loss = nn.MSELoss
name = f"model_{sizes}_output_{output_activation.__name__}_intermediate_{intermediate_activation.__name__}_normalized_data_{normalize_output}_loss_{loss.__name__}"

comparison_table = pd.read_csv("results/comparison_table.csv")

#probabilities = df["weight"]


weight_path = f"results/{name}.pth" if os.path.exists(f"results/{name}.pth") else None
logger = Logger("results/logs")

train_df, val_df, test_df = stratified_split(df)

train_data = CustomDataset(train_df, normalize_output)
val_data = CustomDataset(val_df, normalize_output)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

model = Model(weight_path,description=name, hidden_sizes=sizes, output_activation=output_activation)  
criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, 30, logger)

torch.save(trained_model.state_dict(), f"results/{name}.pth")

new_row = {
    "name": name,
    "loss": val_loss
}

comparison_table = pd.concat([comparison_table, pd.DataFrame([new_row])], ignore_index=True)
comparison_table.to_csv("results/comparison_table.csv", index=False)