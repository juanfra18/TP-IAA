import pandas as pd
from model.eval import evaluate_model
from model.loader import CustomDataset
from model.model import Model
from model.trainer import Trainer
from torch.utils.data import DataLoader
from analysis.data_split import stratified_split
from analysis.logger import Logger


import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os


# Hyperparams
EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.001

df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
sizes = [len(df.columns) - 1, 32, 16, 1]
output_activation = nn.Identity
intermediate_activation = nn.ReLU
normalize_output = True
loss = nn.MSELoss
name = f"model_{sizes}_output_{output_activation.__name__}_intermediate_{intermediate_activation.__name__}_normalized_data_{normalize_output}_loss_{loss.__name__}"

comparison_table = pd.read_csv("results/comparison_table.csv")


weight_path = f"results/{name}.pth" if os.path.exists(f"results/{name}.pth") else None
logger = Logger("results/logs")

train_df, val_df, test_df = stratified_split(df)

train_data = CustomDataset(train_df, normalize_output)
val_data = CustomDataset(val_df, normalize_output)
test_data = CustomDataset(test_df, normalize_output)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = Model(
    weight_path,
    description=name,
    input_size=sizes[0],
    hidden_sizes=sizes,
    output_activation=output_activation,
    hidden_activation=intermediate_activation,
    dropout_p=0.1
)
criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

trained_model, val_loss = Trainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=EPOCHS,
    logger=logger
).train_model(model, optimizer, np.ones(len(train_data.get_column_names()), dtype=int).tolist(),True,True,True,True)

torch.save(trained_model.state_dict(), f"results/{name}.pth")

new_row = {"name": name, "loss": val_loss}


# Evaluate on test
test_data = CustomDataset(test_df, normalize_output)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

preds, trues, metrics = evaluate_model(trained_model, test_loader, normalize_output)

new_row.update(
    {"test_mse": metrics["mse"], "test_mae": metrics["mae"], "test_r2": metrics["r2"]}
)
comparison_table = pd.concat(
    [comparison_table, pd.DataFrame([new_row])], ignore_index=True
)
comparison_table.to_csv("results/comparison_table.csv", index=False)


logger.log(
    f"Test metrics for {name}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}"
)
