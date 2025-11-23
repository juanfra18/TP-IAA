"""
Improved Training Script with 7 Enhancements

Improvements implemented:
1. Remove highly correlated features (moderate subset - 31 features)
2. Switch to L1Loss (MAE loss)
3. Add Batch Normalization
4. Increase epochs to 100
5. Deeper architecture [31, 64, 32, 16, 1]
6. Learning rate scheduler (ReduceLROnPlateau)
7. K-Fold Cross-Validation

Usage:
    python3 train_improved.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from model.eval import evaluate_model
from model.loader import CustomDataset
from model.model import Model
from model.trainer import Trainer
from analysis.data_split import stratified_split
from analysis.logger import Logger


# ============================================================================
# Configuration
# ============================================================================

# Improvement 1: Remove highly correlated features
FEATURES_TO_DROP = ['less_work', 'lost_work']  # Moderate subset

# Improvement 2: L1Loss instead of MSELoss
LOSS_FUNCTION = nn.L1Loss

# Improvement 3: Enable Batch Normalization
USE_BATCH_NORM = True

# Improvement 4: Increase epochs
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 50

# Improvement 5: Deeper architecture
# Will be calculated after dropping features: [num_features, 64, 32, 16, 1]
HIDDEN_LAYERS = [64, 32, 16, 1]

# Other hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_P = 0.2
NORMALIZE_OUTPUT = False

# Improvement 7: K-Fold CV settings
USE_KFOLD = True
N_FOLDS = 5
RANDOM_SEED = 42


# ============================================================================
# Helper Functions
# ============================================================================

def train_single_model(train_df, val_df, test_df, model_description, logger, fold_num=None):
    """Train a single model with the improved configuration."""
    
    # Create datasets with input scaling
    train_data = CustomDataset(train_df, NORMALIZE_OUTPUT, scaler=None, fit_scaler=True)
    # Use the fitted scaler from training data for val and test
    val_data = CustomDataset(val_df, NORMALIZE_OUTPUT, scaler=train_data.scaler, fit_scaler=False)
    test_data = CustomDataset(test_df, NORMALIZE_OUTPUT, scaler=train_data.scaler, fit_scaler=False)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate input size
    input_size = len(train_data.get_column_names())
    sizes = [input_size] + HIDDEN_LAYERS
    
    # Create model
    model = Model(
        weight_path=None,
        description=model_description,
        input_size=input_size,
        hidden_sizes=sizes,
        output_activation=nn.Identity,
        hidden_activation=nn.ReLU,
        dropout_p=DROPOUT_P,
        use_batch_norm=USE_BATCH_NORM
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Improvement 6: Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Create trainer with custom trainer class that supports scheduler
    criterion = LOSS_FUNCTION()
    trainer = ImprovedTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        logger=logger
    )
    
    # Train model
    trained_model, best_val_loss = trainer.train_model(
        model, optimizer, 
        mask=np.ones(input_size, dtype=int).tolist(),
        log_basic=True,
        plot_losses=True,
        save_plots=True,
        log_during_training=False  # Less verbose
    )
    
    # Evaluate on test set
    preds, trues, metrics = evaluate_model(trained_model, test_loader, NORMALIZE_OUTPUT)
    
    # Save model if fold number provided
    if fold_num is not None:
        model_path = f"results/improved_model_fold{fold_num}.pth"
        torch.save(trained_model.state_dict(), model_path)
        logger.log(f"Model saved to {model_path}")
    
    return trained_model, metrics


def train_with_kfold(df, logger):
    """Train model using K-Fold Cross-Validation."""
    
    logger.log(f"\n{'='*60}")
    logger.log(f"Starting {N_FOLDS}-Fold Cross-Validation")
    logger.log(f"{'='*60}")
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_metrics = []
    
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(df), 1):
        logger.log(f"\n--- Fold {fold}/{N_FOLDS} ---")
        
        # Split train_val into train and validation
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]
        
        # Further split train_val into train (80%) and val (20%)
        val_size = int(0.2 * len(train_val_df))
        val_df = train_val_df.iloc[:val_size]
        train_df = train_val_df.iloc[val_size:]
        
        logger.log(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Train model
        model_desc = f"improved_model_fold{fold}"
        model, metrics = train_single_model(train_df, val_df, test_df, model_desc, logger, fold_num=fold)
        
        fold_metrics.append(metrics)
        
        logger.log(f"Fold {fold} Test MAE: {metrics['mae']:.4f} ± {metrics['mae_std']:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate average metrics across folds
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in fold_metrics]),
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'mae_std': np.mean([m['mae_std'] for m in fold_metrics]),
        'r2': np.mean([m['r2'] for m in fold_metrics])
    }
    
    std_metrics = {
        'mse': np.std([m['mse'] for m in fold_metrics]),
        'mae': np.std([m['mae'] for m in fold_metrics]),
        'r2': np.std([m['r2'] for m in fold_metrics])
    }
    
    logger.log(f"\n{'='*60}")
    logger.log(f"K-Fold Cross-Validation Results")
    logger.log(f"{'='*60}")
    logger.log(f"Average MSE: {avg_metrics['mse']:.4f} ± {std_metrics['mse']:.4f}")
    logger.log(f"Average MAE: {avg_metrics['mae']:.4f} ± {std_metrics['mae']:.4f}")
    logger.log(f"Average MAE Std: {avg_metrics['mae_std']:.4f}")
    logger.log(f"Average R²: {avg_metrics['r2']:.4f} ± {std_metrics['r2']:.4f}")
    logger.log(f"{'='*60}\n")
    
    return avg_metrics, std_metrics, fold_metrics


# ============================================================================
# Improved Trainer with LR Scheduler
# ============================================================================

class ImprovedTrainer(Trainer):
    """Extended Trainer class with learning rate scheduler support."""
    
    def __init__(self, train_loader, val_loader, test_loader, criterion, 
                 optimizer, scheduler, num_epochs, patience, logger):
        super().__init__(train_loader, val_loader, test_loader, criterion, 
                        optimizer, num_epochs, logger)
        self.scheduler = scheduler
        self.patience = patience
    
    def train_model(self, model, optimizer, mask, log_basic=True, 
                   plot_losses=True, save_plots=True, log_during_training=True):
        """Train model with learning rate scheduling."""
        
        model.train()
        
        if log_basic:
            self.logger.log(f"Starting training on model: {model.description}")
        
        best_val_loss = float('inf')
        best_model_state = model.state_dict()
        no_improve_epochs = 0
        
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            epoch_train_loss = self.train_epoch(model, optimizer, mask_tensor)
            train_losses.append(epoch_train_loss)
            
            epoch_val_loss = self.validate_epoch(model, mask_tensor)
            val_losses.append(epoch_val_loss)
            
            # Update learning rate scheduler
            self.scheduler.step(epoch_val_loss)
            
            if log_during_training and (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.log(
                    f'Epoch {epoch+1}/{self.num_epochs}, '
                    f'Train Loss: {epoch_train_loss:.4f}, '
                    f'Val Loss: {epoch_val_loss:.4f}, '
                    f'LR: {current_lr:.6f}'
                )
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.patience:
                    if log_basic:
                        self.logger.log(f"Early stopping at epoch {epoch+1}/{self.num_epochs}")
                    model.load_state_dict(best_model_state)
                    break
        
        if log_basic:
            self.logger.log(f"Training complete, best validation loss: {best_val_loss:.4f}")
        
        if plot_losses:
            self.plot_train_losses(train_losses, val_losses, model)
        
        return model, best_val_loss


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Setup logger
    logger = Logger("results/logs")
    
    logger.log("\n" + "="*60)
    logger.log("IMPROVED MODEL TRAINING")
    logger.log("="*60)
    logger.log("\nImprovements Applied:")
    logger.log("1. ✓ Removed highly correlated features (moderate subset)")
    logger.log(f"   Dropped: {FEATURES_TO_DROP}")
    logger.log(f"2. ✓ Switched to L1Loss (MAE loss)")
    logger.log(f"3. ✓ Added Batch Normalization: {USE_BATCH_NORM}")
    logger.log(f"4. ✓ Increased epochs to {EPOCHS}")
    logger.log(f"5. ✓ Deeper architecture: {HIDDEN_LAYERS}")
    logger.log(f"6. ✓ Learning rate scheduler: ReduceLROnPlateau")
    logger.log(f"7. ✓ K-Fold Cross-Validation: {N_FOLDS} folds")
    logger.log("="*60 + "\n")
    
    # Load and preprocess data
    logger.log("Loading data...")
    df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
    logger.log(f"Original dataset: {df.shape}")
    
    # Improvement 1: Drop highly correlated features
    df = df.drop(columns=FEATURES_TO_DROP)
    logger.log(f"After dropping correlated features: {df.shape}")
    logger.log(f"Dropped features: {FEATURES_TO_DROP}\n")
    
    if USE_KFOLD:
        # Train with K-Fold CV
        avg_metrics, std_metrics, fold_metrics = train_with_kfold(df, logger)
        
        # Save results
        results_df = pd.DataFrame(fold_metrics)
        results_df.to_csv("results/kfold_results.csv", index=False)
        logger.log("Results saved to results/kfold_results.csv")
        
        # Save summary
        summary = pd.DataFrame([{
            'model': 'improved_model_kfold',
            'avg_mae': avg_metrics['mae'],
            'std_mae': std_metrics['mae'],
            'avg_mae_std': avg_metrics['mae_std'],
            'avg_r2': avg_metrics['r2'],
            'std_r2': std_metrics['r2'],
            'n_folds': N_FOLDS,
            'features_dropped': ', '.join(FEATURES_TO_DROP),
            'batch_norm': USE_BATCH_NORM,
            'architecture': str(HIDDEN_LAYERS)
        }])
        summary.to_csv("results/improved_model_summary.csv", index=False)
        
    else:
        # Train single model with regular train/val/test split
        logger.log("Using single train/val/test split...\n")
        train_df, val_df, test_df = stratified_split(df)
        
        model_desc = f"improved_model_{'_'.join(map(str, HIDDEN_LAYERS))}_bn{USE_BATCH_NORM}"
        model, metrics = train_single_model(train_df, val_df, test_df, model_desc, logger)
        
        logger.log(f"\nFinal Test Metrics:")
        logger.log(f"MSE: {metrics['mse']:.4f}")
        logger.log(f"MAE: {metrics['mae']:.4f} ± {metrics['mae_std']:.4f}")
        logger.log(f"R²: {metrics['r2']:.4f}")
        
        # Save model
        torch.save(model.state_dict(), f"results/{model_desc}.pth")
        logger.log(f"\nModel saved to results/{model_desc}.pth")


if __name__ == "__main__":
    main()
