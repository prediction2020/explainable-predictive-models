"""
File name: dataset.py
Author: Esra Zihni
Date created: 21.05.2018

This file contains the ClinicalDataset class object that is used for working 
with a given dataset.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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

    def __init__(self, name: str, path: str):
        self.name = name
        self.load_data(path)

    def load_data(self, path: str) -> None:
        """
        Loads data from given path.
        :param path: Path to dataset file.
        """
        try:
            self.df = pd.read_pickle(path)
        except IOError:
            print("Wrong file format. Please provide a .pkl file")
        self.cols = list(self.df)
        self.label = list(self.df)[-1]
        self.cat_data = self.df.columns[self.df.dtypes == "category"]
        self.num_data = list(set(self.cols) - set(self.cat_data))
        self.X, self.y = self.df.drop(self.label, axis=1), self.df[[self.label]]
        self.preds = list(self.X)
        self.cat_preds = self.X.columns[self.X.dtypes == "category"]


    def assign_train_test_splits(
        self, path: str, test_size: float = None, splits: int = None
    ) -> None:
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
            self.splits = pickle.load(open(path, "rb"))
            print("Loaded training and test splits from pre-existing splits.")

        except IOError:
            self.splits = []
            for i in range(splits):
                # Creating the same test data split to use in all models
                tmp_split = dict.fromkeys(
                    ["train_data", "train_labels", "test_data", "test_labels"]
                )
                X_tr, X_te, y_tr, y_te = train_test_split(
                    self.X, self.y, test_size=test_size, stratify=self.y
                )

                # Save the splits to be used in all models
                tmp_split["train_data"] = X_tr
                tmp_split["train_labels"] = y_tr
                tmp_split["test_data"] = X_te
                tmp_split["test_labels"] = y_te

                self.splits.append(tmp_split)

            pickle.dump(self.splits, open(path, "wb"))

    def subsample_training_sets(
        self, number_of_splits: int, subsampling_type: str
    ) -> None:
        """
        Subsamples the training set according to the defined subsampling method.

        :param number_of_splits: Number of training-test splits.
        :param subsampling_tpye: Subsampling method to be used.
        """
        for i in range(number_of_splits):
            self.splits[i]["train_data"], self.splits[i]["train_labels"] = subsample(
                self.splits[i]["train_data"],
                self.splits[i]["train_labels"],
                subsampling_type,
            )

    def impute(self, number_of_splits: int, imputation_type: str = "mean/mode")-> None:
        """
        
        """
        for i in range(number_of_splits):
            train = self.splits[i]["train_data"].copy()
            test = self.splits[i]["test_data"].copy()

            if imputation_type == "mean/mode":
                # Calculate the mean of numerical variables in the dataset and round the 
                # floating point to two.
                num_data_means_training = round(train.loc[:, self.num_data].mean(),2)
                # Calculate the mode of categorical data in the dataset
                cat_data_modes_training = round(train.loc[:, self.cat_preds].mean())

                train.fillna(pd.concat((num_data_means_training,cat_data_modes_training)), inplace = True)
                test.fillna(pd.concat((num_data_means_training,cat_data_modes_training)), inplace = True)

                self.splits[i]["train_data"] = train
                self.splits[i]["test_data"]= test
            

    def normalize(self, number_of_splits: int)-> None:
        """
        Scales the continuous data in the dataset.
        """
        # Center numeric data
        for i in range(number_of_splits):
            train = self.splits[i]["train_data"].loc[:, self.num_data].copy()
            test = self.splits[i]["test_data"].loc[:, self.num_data].copy()

            num_data_means_training = train.mean()
            num_data_stds_training = train.std()

            scaled_train = (train - num_data_means_training) / num_data_stds_training
            scaled_test = (test - num_data_means_training) / num_data_stds_training

            self.splits[i]["train_data"].loc[:, self.num_data] = scaled_train
            self.splits[i]["test_data"].loc[:, self.num_data] = scaled_test

