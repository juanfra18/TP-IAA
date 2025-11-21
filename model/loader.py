import pandas as pd
import torch
from analysis.data_processer import DataProcesser

class CustomDataset(torch.utils.data.Dataset):
    
    data : pd.DataFrame
    column_count : int
    
    def __init__(self, df : pd.DataFrame, normalize_output: bool) -> None:
        self.data =  df
        self.column_count = len(self.data.columns) - 1
        
        self.labels = self.data["target"].astype('float64')
        self.data = self.data.drop(columns=["target"])
        
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
