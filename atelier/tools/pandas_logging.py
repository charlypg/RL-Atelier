import pandas as pd
from pathlib import Path

class PandasLogger:
    def __init__(self):
        self.df = pd.DataFrame()
    
    @property
    def data(self):
        return self.df
    
    def __str__(self):
        return self.df.__str__()
    
    def __repr__(self):
        return self.df.__repr__()
    
    def log(self, data: dict):
        if type(data) != dict:
            raise ValueError("log() requires a dict")

        if self.df.empty and len(self.df.columns) == 0:
            self.df = pd.DataFrame(columns=data.keys())
        
        self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
    
    def save(self, filepath: str):
        """
        Save internal dataframe to a CSV file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(filepath, index=False)
    
    def load(self, filepath: str):
        """
        Load data from a CSV file into the internal dataframe.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No file found at {filepath}")

        self.df = pd.read_csv(filepath)
