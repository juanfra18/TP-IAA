import torch
import torch.nn as nn

def metrics(y_true, y_pred): 
    mse = nn.MSELoss()(y_true, y_pred).item()
    rmse = mse ** 0.5
    mae = nn.L1Loss()(y_true, y_pred).item()

    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    smape = torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)
    smape = 100 * torch.mean(smape).item() #actualmente se calcula sobre los valores normalizados, tal vez conviene desnormalizarlo antes

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "smape": smape
    }