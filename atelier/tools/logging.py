from abc import ABC
import os
import csv
import pandas as pd
from pathlib import Path


class Logger(ABC):
    def log(self, dico: dict, step: int):
        raise NotImplementedError


class StrLogger(Logger):
    def __init__(self, debug: bool = False):
        self.debug = debug

    def log(self, dico, step) -> str:
        string = str()
        string += "step: {} ; ".format(step)
        string += "sum_rewards_median: {:.3f} ; ".format(dico["sum_rewards_median"])
        string += "sum_rewards_25: {:.3f} ; ".format(dico["sum_rewards_25"])
        string += "sum_rewards_75: {:.3f} ; ".format(dico["sum_rewards_75"])
        if self.debug:
            exclude = {"step", "sum_rewards_median", "sum_rewards_25", "sum_rewards_75"}
            for k, v in zip(dico.keys(), dico.values()):
                if not(k in exclude):
                    string += "{0}: {1} ; ".format(k, v)
        return string


class PrintLogger(StrLogger):
    def log(self, dico, step):
        print(StrLogger.log(self, dico, step))


def append_step(dico: dict, step: int):
    dico_to_write = dico.copy()
    dico_to_write["step"] = step
    return dico_to_write

class CSVLogger(Logger):
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
