import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

from analysis.logger import Logger
from model.model import Model


class Trainer:
    
    def __init__(self, train_loader : torch.utils.data.DataLoader, val_loader : torch.utils.data.DataLoader, test_loader : torch.utils.data.DataLoader, criterion : nn.Module, optimizer : torch.optim.Optimizer, num_epochs : int, logger : Logger) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.logger = logger


    def plot_train_losses(self, train_losses : list[float], val_losses : list[float], model : nn.Module) -> None:
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/train_val_loss_{model.description}.png')
        
    def train_model_from_scratch(self, model : Model, mask : list[int], log_basic : bool = True, plot_losses : bool = True, save_plots : bool = True, log_during_training : bool = True) -> tuple[nn.Module, float]:
        model = Model(
            weight_path=None,
            description=model.description,
            input_size=model.input_size,
            hidden_sizes=model.hidden_sizes,
            hidden_activation=model.hidden_activation,
            output_activation=model.output_activation,
            dropout_p=model.dropout_p
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return self.train_model(model, optimizer, mask, log_basic, plot_losses, save_plots, log_during_training)
        
        

    def train_model(self, model : nn.Module, optimizer : torch.optim.Optimizer, mask : list[int], log_basic, plot_losses : bool = True, save_plots : bool = True, log_during_training : bool = True) -> tuple[nn.Module, float]:
        
        model.train()
        
        if log_basic:
            self.logger.log(f"Starting training on model: {model.description}")
            self.logger.log(f"Mask: {mask}")
        
        best_val_loss = float('inf')
        best_model_state = model.state_dict()
        patience = 50
        no_improve_epochs = 0
        
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            
            epoch_train_loss = self.train_epoch(model, optimizer, mask_tensor)
            train_losses.append(epoch_train_loss)


            epoch_val_loss = self.validate_epoch(model, mask_tensor)
            val_losses.append(epoch_val_loss)

            if log_during_training:
                self.logger.log(
                    f'Epoch {epoch+1}/{self.num_epochs}, '
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
                    if log_basic:
                        self.logger.log(f"Early stopping at epoch {epoch+1}/{self.num_epochs}")
                    model.load_state_dict(best_model_state)
                    break

        if log_basic:
            self.logger.log(f"Training complete, best validation loss: {best_val_loss:.8f}")
            self.logger.log(f"Mask: {mask}")
        
        if plot_losses:
            self.plot_train_losses(train_losses, val_losses, model)
        if save_plots:
            plt.savefig(f'results/plots/train_val_loss_{model.description}.png')
        
        return model, best_val_loss
    
    def train_epoch(self, model : nn.Module, optimizer : torch.optim.Optimizer, mask_tensor : torch.Tensor) -> float:
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in self.train_loader:
            optimizer.zero_grad()
            outputs = model(inputs * mask_tensor)
            loss = self.criterion(outputs, labels)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        return epoch_train_loss / len(self.train_loader)
    
    def validate_epoch(self, model : nn.Module, mask_tensor : torch.Tensor) -> float:
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = model(inputs * mask_tensor)
                loss = self.criterion(outputs, labels)
                epoch_val_loss += loss.item()
        return epoch_val_loss / len(self.val_loader)
    
    def test_model(self, model : nn.Module, mask_tensor : torch.Tensor) -> float:
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = model(inputs * mask_tensor)
                loss = self.criterion(outputs, labels)
                epoch_test_loss += loss.item()
        return epoch_test_loss / len(self.test_loader)
