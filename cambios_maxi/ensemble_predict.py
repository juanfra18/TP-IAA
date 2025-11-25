"""
Ensemble Prediction Script

This script loads all 5 models trained during K-fold cross-validation
and creates an ensemble by averaging their predictions.

Expected improvement: -0.2 to -0.4 MAE reduction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import Model
from model.loader import CustomDataset
from analysis.data_split import stratified_split
from analysis.logger import Logger


FEATURES_TO_DROP = ['less_work', 'lost_work']
HIDDEN_LAYERS = [64, 32, 16, 1]
USE_BATCH_NORM = True
DROPOUT_P = 0.2
NORMALIZE_OUTPUT = False
BATCH_SIZE = 32


def load_fold_model(fold_num, input_size):
    """Load a trained model from a specific fold."""
    model_path = f"improved_model_fold{fold_num}.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    sizes = [input_size] + HIDDEN_LAYERS
    model = Model(
        weight_path=None,
        description=f"improved_model_fold{fold_num}",
        input_size=input_size,
        hidden_sizes=sizes,
        output_activation=nn.Identity,
        hidden_activation=nn.ReLU,
        dropout_p=DROPOUT_P,
        use_batch_norm=USE_BATCH_NORM
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model


def get_model_predictions(model, data_loader):
    """Get predictions from a single model."""
    preds = []
    trues = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    
    return np.array(preds), np.array(trues)


def ensemble_predict(models, data_loader):
    """Get ensemble predictions by averaging all models."""
    all_predictions = []
    ground_truth = None
    
    for model in models:
        preds, trues = get_model_predictions(model, data_loader)
        all_predictions.append(preds)
        if ground_truth is None:
            ground_truth = trues
    
    # Average predictions across all models
    ensemble_preds = np.mean(all_predictions, axis=0)
    
    return ensemble_preds, ground_truth, all_predictions


def evaluate_predictions(preds, trues, description="Model"):
    """Evaluate predictions and return metrics."""
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    
    # Calculate MAE std
    absolute_errors = np.abs(trues - preds)
    mae_std = np.std(absolute_errors)
    
    return {
        'description': description,
        'mse': mse,
        'mae': mae,
        'mae_std': mae_std,
        'r2': r2
    }


def main():
    logger = Logger("results/logs")
    
    logger.log("\n" + "="*60)
    logger.log("ENSEMBLE PREDICTION - Averaging 5 Fold Models")
    logger.log("="*60)
    
    logger.log("\nLoading data...")
    df = pd.read_csv("../datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
    df = df.drop(columns=FEATURES_TO_DROP)
    logger.log(f"Dataset shape after dropping features: {df.shape}")
    
    train_df, val_df, test_df = stratified_split(df)
    logger.log(f"Test set size: {len(test_df)}")
    
    test_data = CustomDataset(test_df, NORMALIZE_OUTPUT, scaler=None, fit_scaler=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = len(test_data.get_column_names())
    logger.log(f"Input features: {input_size}")
    
    logger.log("\nLoading 5 fold models...")
    models = []
    for fold in range(1, 6):
        try:
            model = load_fold_model(fold, input_size)
            models.append(model)
            logger.log(f"✓ Loaded fold {fold} model")
        except FileNotFoundError as e:
            logger.log(f"✗ {e}")
            return
    
    logger.log(f"\nSuccessfully loaded {len(models)} models")
    
    logger.log("\n" + "-"*60)
    logger.log("Individual Model Performance on Test Set:")
    logger.log("-"*60)
    
    individual_results = []
    for i, model in enumerate(models, 1):
        preds, trues = get_model_predictions(model, test_loader)
        metrics = evaluate_predictions(preds, trues, f"Fold {i}")
        individual_results.append(metrics)
        logger.log(f"Fold {i}: MAE={metrics['mae']:.4f} ± {metrics['mae_std']:.4f}, R²={metrics['r2']:.4f}")
    
    # Calculate ensemble predictions
    logger.log("\n" + "-"*60)
    logger.log("Computing Ensemble Predictions...")
    logger.log("-"*60)
    
    ensemble_preds, ground_truth, all_preds = ensemble_predict(models, test_loader)
    ensemble_metrics = evaluate_predictions(ensemble_preds, ground_truth, "Ensemble")
    
    logger.log(f"\nENSEMBLE RESULTS:")
    logger.log(f"   MSE: {ensemble_metrics['mse']:.4f}")
    logger.log(f"   MAE: {ensemble_metrics['mae']:.4f} ± {ensemble_metrics['mae_std']:.4f}")
    logger.log(f"   R²:  {ensemble_metrics['r2']:.4f}")
    
    avg_individual_mae = np.mean([m['mae'] for m in individual_results])
    std_individual_mae = np.std([m['mae'] for m in individual_results])
    
    logger.log("\n" + "="*60)
    logger.log("COMPARISON")
    logger.log("="*60)
    logger.log(f"Average Individual MAE: {avg_individual_mae:.4f} ± {std_individual_mae:.4f}")
    logger.log(f"Ensemble MAE:          {ensemble_metrics['mae']:.4f}")
    logger.log(f"Improvement:           {avg_individual_mae - ensemble_metrics['mae']:.4f} MAE reduction")
    logger.log("="*60 + "\n")
    
    results_df = pd.DataFrame(individual_results + [ensemble_metrics])
    results_df.to_csv("results/ensemble_comparison.csv", index=False)
    logger.log("Results saved to results/ensemble_comparison.csv")
    
    predictions_df = pd.DataFrame({
        'true': ground_truth,
        'ensemble_pred': ensemble_preds,
        'absolute_error': np.abs(ground_truth - ensemble_preds)
    })
    
    for i, preds in enumerate(all_preds, 1):
        predictions_df[f'fold{i}_pred'] = preds
    
    predictions_df.to_csv("results/ensemble_predictions.csv", index=False)
    logger.log("Predictions saved to results/ensemble_predictions.csv")
    
    pred_variance = np.var(all_preds, axis=0)
    avg_variance = np.mean(pred_variance)
    logger.log(f"\nModel agreement (avg variance): {avg_variance:.4f}")
    logger.log("(Lower variance = models agree more)")


if __name__ == "__main__":
    main()
