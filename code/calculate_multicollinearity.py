"""
File name: calculate_multicollinearity.py
Author: Esra Zihni
Date created: 15.01.2018

This script performs multicollinearity analysis on the clinical predictors. It 
reads the path options from the config.yml file. It loads the dataset and 
calculates a Variance Inflation Factor (VIF) for each predictor to measure
its multicollinearity. It saves the calculated VIF values into a csv file.
"""

import os

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing

from utils.dataset import ClinicalDataset

########################################################################################
###### ASSIGN CONFIGURATION VARIABLES ##################################################
########################################################################################

# You need to add constructor in order to be able to use the join command in the yaml
# file
def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join(str(i) for i in seq)


yaml.add_constructor("!join", join)

# Read the config file
cfg = yaml.load(open("config.yml", "r"))

# Assign variables
dataset_name = cfg["dataset name"]
dataset_path = cfg["data path"]
imputation_type = cfg["imputation type"]
to_save_path = "data_analysis/multicollinearity_analysis/"


#########################################################################################
###### CALCULATE AND SAVE VARIANCE INFLATION FACTORS ####################################
#########################################################################################

data = ClinicalDataset(name=dataset_name, path=dataset_path)
df = data.X

# Impute missing values
if imputation_type == "mean/mode":
    # Calculate the mean of numerical variables in the dataset and round the 
    # floating point to two.
    num_data_means = round(df.loc[:, data.num_data].mean(),2)
    # Calculate the mode of categorical data in the dataset
    cat_data_modes = round(df.loc[:, data.cat_preds].mean())

    df.fillna(pd.concat((num_data_means,cat_data_modes)), inplace = True)   

# Center numeric data
df.loc[:, data.num_data] = preprocessing.StandardScaler().fit_transform(
    df.loc[:, data.num_data]
)

# Get names of predictors
predictor_names = data.preds

# First convert all data types to float64 - this is needed in order to use 
# the variance_inflation_factor function
for i, col in enumerate(predictor_names):
    df[col] = df[col].astype("float64")

# Calculate VIF values for each predictor
vif = np.array([variance_inflation_factor(df.values, i) for i in range(df.shape[1])])

# Save VIFs to a pandas DataFrame
df_vif = pd.DataFrame(
    vif, index=predictor_names, columns=["Variance Inflation Factor(VIF)"]
)


# Create folder to save results if it doesn't exist
if not os.path.exists(to_save_path):
    os.makedirs(to_save_path)

# Save values
df_vif.to_csv(
    f"{to_save_path}{dataset_name}_multicollinearity.csv", float_format="%2.2f"
)
