# TODO : receive loaded data and clean it . the clean process should handle discretization, handle missing data,
# TODO: normalaization and so on .
# TODO: return the clean data
import pandas as pd

class data_processing():
    def __init__(self,dataset_path,sep):
        self.dataset = pd.read_csv(dataset_path,sep=sep)

    def prepare_data(self):
        self.dataset.dropna(inplace=True)
        self

