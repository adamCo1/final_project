import sklearn.metrics
import pandas as pd
from sklearn.datasets import load_boston


'''
@Params dataset: a pandas DataFrame
        feature_set: a list of features names
'''
class Mutual_Information_Estimator():

    def __init__(self, dataset):
        self.dataset = dataset
        self.features_vectors = self.dataset.loc[:,"0":"29"]
        self.class_vector = self.dataset.loc[:,"30"]
        self.features_names = dataset.columns

    def calculate_score(self,features_names):
        total_score = 0
        for feature in features_names:
            total_score = total_score + sklearn.metrics.mutual_info_score(self.class_vector, self.features_vectors[feature], contingency=None)
            print(total_score)
        return total_score


wdbc_df = pd.read_csv("wdbc.csv",sep = ",")
mi = Mutual_Information_Estimator(wdbc_df)
features_name = ['1','5','4','23']
features = mi.dataset.loc[:,"0":"29"]
classes = mi.dataset.loc[:,"30"]
score = mi.calculate_score(features_name)
print(mi.dataset.dtypes)
print(score)