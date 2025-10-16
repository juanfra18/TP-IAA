import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from analysis.data_processer import DataProcesser
from torch.utils.data import DataLoader
import torch.nn as nn
import os



def plot_evaluation_results(trues, preds, name):
    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, alpha=0.6)
    plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.savefig(f"results/plots/predicted_vs_actual_{name}.png", bbox_inches="tight")
    plt.close()

    residuals = trues - preds
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30, edgecolor="k")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(f"Residuals Histogram - {name}")
    plt.grid(True)
    plt.savefig(f"results/plots/residuals_{name}.png", bbox_inches="tight")
    plt.close()


def evaluate_model(model: nn.Module, loader: DataLoader, normalize_output: bool):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            # ensure tensors are on cpu before converting
            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)
    if normalize_output:
        preds = preds * DataProcesser.MAX_SCORE
        trues = trues * DataProcesser.MAX_SCORE

    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    plot_evaluation_results(trues, preds, model.description)
    return preds, trues, {"mse": float(mse), "mae": float(mae), "r2": float(r2)}


