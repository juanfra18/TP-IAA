import pandas as pd
import torch
from analysis.data_processer import DataProcesser
from sklearn.preprocessing import StandardScaler
import pickle
import os

class CustomDataset(torch.utils.data.Dataset):
    
    data : pd.DataFrame
    column_count : int
    
    def __init__(self, df : pd.DataFrame, normalize_output: bool = False, scaler=None, fit_scaler: bool = True) -> None:
        self.data =  df
        self.column_count = len(self.data.columns) - 1
        
        self.labels = self.data["target"].astype('float64')
        self.data = self.data.drop(columns=["target"])
        
        # Apply input feature scaling
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            self.data = pd.DataFrame(
                self.scaler.fit_transform(self.data),
                columns=self.data.columns,
                index=self.data.index
            )
        elif scaler is not None:
            self.scaler = scaler
            self.data = pd.DataFrame(
                self.scaler.transform(self.data),
                columns=self.data.columns,
                index=self.data.index
            )
        else:
            self.scaler = None
        
        if normalize_output:
            self.labels = self.labels.astype('float64') / DataProcesser.MAX_SCORE
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_label = self.labels.iloc[idx]
        sample_data = self.data.iloc[idx].values
        return torch.tensor(sample_data, dtype=torch.float32), torch.tensor(sample_label, dtype=torch.float32)
    
    def get_column_names(self) -> list[str]:
        return self.data.columns.tolist()
