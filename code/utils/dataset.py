"""
File name: dataset.py
Author: Esra Zihni
Date created: 


"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utils.helper_functions import subsample


class ClinicalDataset:
    """
    A class used to represent the clinical data.

    :param name: Name of the dataset.
    :param path: Path to the dataset file.

    .. py:meth:: ClinicalDataset.load_data(path)
        :param path: Path to dataset file.

    .. py:meth:: ClinicalDataset.preprocess()
    
    .. py_meth:: ClinicalDataset.assign_train_test_splits(path, test_size, splits)
        :param path: Path to splits file.
        :param test_size: The proportion of the dataset to include in the test split.
        :param splits: Number of training-test splits, i.e shuffles

    .. py.meth:: ClinicalDataset.subsample(number_of_splits, subsampling_type)
        :param number_of_splits: Number of training-test splits.
        :param subsampling_tpye: Subsampling method to be used.
        
    """
    def __init__(self, name:str, path:str):
        self.name = name
        self.load_data(path)

    def load_data(self,path:str)-> None:
        """
        Loads data from given path.
        :param path: Path to dataset file.
        """
        try:
            self.df = pd.read_pickle(path) 
        except IOError:
            print('Wrong file format. Please provide a .pkl file')    
        self.cols = list(self.df)
        self.label = list(self.df)[-1]
        self.cat_data = self.df.columns[self.df.dtypes=='category']
        self.num_data = list(set(self.cols) - set(self.cat_data))
        self.X, self.y = self.df.drop(self.label,axis=1), self.df[[self.label]]
        self.cat_preds = self.X.columns[self.X.dtypes=='category']

    def preprocess(self):
        """
        Centers the continuous data in the dataset.
        """
    	# First convert all data types to float64 - this is needed in order to use the StandardScaler function
        for i, col in enumerate(self.cols):
            self.df[col] = self.df[col].astype('float64')

        # Center numeric data       
        self.X.loc[:,self.num_data] = preprocessing.StandardScaler().fit_transform(self.X.loc[:,self.num_data])

    def assign_train_test_splits(self, path:str, test_size:float=None, splits:int=None)-> None:
        """
        Splits the dataset into training and test sets repeatedly for the specified
        number of splits. Each run results in different training and test sets originating
        from the same dataset.

        If the splits file already exists, loads pre-existing splits. 

        :param path: Path to splits file.
        :param test_size: The proportion of the dataset to include in the test split.
        :param splits: Number of training-test splits.
        """
        try:
            self.splits = pickle.load(open(path,'rb'))
            print('Loaded training and test splits from pre-existing splits.')

        except IOError:
            self.splits = []
            for i in range(splits):
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

    def subsample_training_set(self,number_of_splits:int,subsampling_type:str)-> None:
        """
        Subsamples the training set according to the defined subsampling method.

        :param number_of_splits: Number of training-test splits.
        :param subsampling_tpye: Subsampling method to be used.
        """
        for i in range(number_of_splits):
            self.splits[i]['train_data'], self.splits[i]['train_labels'] = subsample(self.splits[i]['train_data'], self.splits[i]['train_labels'],subsampling_type)
