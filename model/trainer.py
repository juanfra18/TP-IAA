import torch 
import torch.nn as nn

from analysis.logger import Logger

def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                num_epochs: int, 
                logger : Logger) -> tuple[nn.Module, float]:
    
    model.train()
    
    logger.log(f"Starting training on model: {model.description}")
    
    best_val_loss = float('inf')
    best_model_state = model.state_dict()
    patience = 5
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
             
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset) # pyright: ignore[reportArgumentType]
        logger.log(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_loss:.4f}')
        
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.log(f"Early stopping at epoch {epoch+1}/{num_epochs}")
                model.load_state_dict(best_model_state)  # Restore best model
                break

    
    print('Training complete')
    
    return model, best_val_loss
    
