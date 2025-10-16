import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

from analysis.logger import Logger

def plot_train_losses(model, train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/plots/train_val_loss_{model.description}.png')


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
    patience = 50
    no_improve_epochs = 0
    
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        #---Entrenamiento---#
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()   
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)  # pyright: ignore[reportArgumentType]
        train_losses.append(epoch_train_loss)


        #---Validacion---#
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)  # pyright: ignore[reportArgumentType]
        val_losses.append(epoch_val_loss)

        logger.log(
            f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {epoch_train_loss:.8f}, '
            f'Val Loss: {epoch_val_loss:.8f}'
        )
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.log(f"Early stopping at epoch {epoch+1}/{num_epochs}")
                model.load_state_dict(best_model_state)  # Restore best model
                break

    
    print('Training complete')
    plot_train_losses(model, train_losses, val_losses)
    
    return model, best_val_loss
    
