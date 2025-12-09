import os
import csv
import pandas as pd
from pathlib import Path

def append_step(dico: dict, step: int):
    dico_to_write = dico.copy()
    dico_to_write["step"] = step
    return dico_to_write

class CSVLogger:
    """CSV logger for important curves"""
    def __init__(self, save_dir, filename):
        self.filepath = os.path.join(save_dir, filename)
        self.initialized = False

    def log(self, dico: dict, step: int):
        dico_to_write = append_step(dico, step)
        with open(self.filepath, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dico_to_write.keys())
            if not(self.initialized):
                writer.writeheader()
                self.initialized = True
            writer.writerow(dico_to_write)

    def get_dataframe(self):
        """Loads data from a CSV file into a dataframe."""
        filepath = Path(self.filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No file found at {filepath}")
        return pd.read_csv(filepath)
