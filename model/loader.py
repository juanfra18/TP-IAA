import pandas as pd
import torch
from analysis.data_processer import DataProcesser

class CustomDataset(torch.utils.data.Dataset):
    
    data : pd.DataFrame
    probabilities : pd.Series
    column_count : int
    
    def __init__(self, df : pd.DataFrame, normalize_output: bool) -> None:
        self.data =  df
        self.probabilities = df["probability"]
        self.column_count = len(self.data.columns) - 2  # Exclude 'score' and 'probability' columns
        if normalize_output:
            self.data["score"] = self.data["score"] / DataProcesser.MAX_SCORE
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        df = self.data.sample(n=1, weights=self.probabilities)
        sample_label = df["score"].values[0]
        sample_data = df.drop(columns=["score", "probability"]).values[0]
        return torch.tensor(sample_data, dtype=torch.float32), torch.tensor(sample_label, dtype=torch.float32)