import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utils.helper_functions import subsample


class ClinicalDataset:
    def __init__(self, name, path):
        self.name = name
        self.load_data(path)

    def load_data(self,datapath):
        self.df = pd.read_pickle(datapath)     
        self.cols = list(self.df)
        self.label = list(self.df)[-1]
        self.cats = self.df.columns[self.df.dtypes=='category']
        self.X, self.y = self.df.drop(self.label,axis=1), self.df[[self.label]]
        self.cat_preds = self.X.columns[self.X.dtypes=='category']

    def preprocess(self):
    	# Convert all data types to float64 - this is needed in order to use the StandardScaler function of sklearn
        for i, col in enumerate(self.cols):
            self.df[col] = self.df[col].astype('float64')

        # Center numeric data
        self.num_data = list(set(self.cols) - set(self.cats))
        self.X.loc[:,self.num_data] = preprocessing.StandardScaler().fit_transform(self.X.loc[:,self.num_data])

    def assign_train_test_splits(self, path, test_size=None, runs=None):
        try:
            self.splits = pickle.load(open(path,'rb'))
            print('Loaded training and test splits from pre-existing splits.')

        except IOError:
            self.splits = []
            for i in range(runs):
                # Creating the same test data split to use in all models
                tmp_split = dict.fromkeys(['train_data','train_labels','test_data','test_labels'])
                X_tr, X_te, y_tr, y_te = train_test_split(self.X, 
                                                          self.y, 
                                                          test_size = test_size,
                                                          stratify=self.y)

                # Save the splits to be used in all models
                tmp_split['train_data'] = X_tr
                tmp_split['train_labels'] = y_tr
                tmp_split['test_data'] = X_te
                tmp_split['test_labels'] = y_te

                self.splits.append(tmp_split)

            pickle.dump(self.splits, open(path, 'wb'))

    def subsample_training_set(self,number_of_runs,subsampling_type):
        for i in range(number_of_runs):
            self.splits[i]['train_data'], self.splits[i]['train_labels'] = subsample(self.splits[i]['train_data'], self.splits[i]['train_labels'],subsampling_type)
