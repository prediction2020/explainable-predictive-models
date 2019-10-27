import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from subsampling import *


class clinical_dataset:
    def __init__(self, name):
        self.name = name
        self.load_data()

    def load_data(self,datasource):
        self.df = pd.read_pickle(datasource)

        self.cats = self.df.columns[self.df.dtypes=='category']
        self.cols = list(self.df)
        self.label = list(self.df)[-1]

        self.X, self.y = self.df.drop(self.label,axis=1), self.df[[self.label]]

    def preprocess(self):
    	# Convert all data types to float64 - this is needed in order to use the StandardScaler function of sklearn
        for i, col in enumerate(self.cols):
            self.df[col] = self.df[col].astype('float64')

        # Center numeric data
        self.num_data = list(set(self.cols) - set(self.cats))
        self.X.loc[:,self.num_data] = preprocessing.StandardScaler().fit_transform(self.X.loc[:,self.num_data])

    def assign_training_test_sets(self, test_size=None, runs=None):
        self.sets = []
        for i in range(runs):
            # Creating the same test data split to use in all models
            tmp_set = dict.fromkeys(['train_data','train_labels','test_data','test_labels'])
            X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size = test_size,stratify=self.y, random_state = 42+i)

            # Save the sets to be used in all models
            tmp_set['train_data'] = X_tr
            tmp_set['train_labels'] = y_tr
            tmp_set['test_data'] = X_te
            tmp_set['test_labels'] = y_te

            self.sets.append(tmp_set)

    def subsample_training_set(self,number_of_runs,subsampling_type):
        for i in range(number_of_runs):
            self.sets[i]['train_data'], self.sets[i]['train_labels'] = subsample(self.sets[i]['train_data'], self.sets[i]['train_labels'],subsampling_type)
