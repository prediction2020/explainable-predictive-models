"""
File name: evaluate_feature_importance.py
Author: Esra Zihni
Date created: 10.08.2018

This is the main script to evaluate the importance of predictors. It reads the 
implementation and path options from the config.yml script. It loads the training and 
test sets, loads the specified trained model(s) and calculates features importance 
scores using the model-specific explainability methods. Finally, it saves the calculated 
feature importance scores as csv files.
"""

import os

import numpy as np
import pandas as pd
import yaml

from utils.dataset import *
from utils.helper_functions import calc_deep_taylor_values, calc_shap_values, calc_kernel_shap_values,config_handler,linear_shap_value 
from utils.models import *

def evaluate_feature_importance():

    ########################################################################################
    #### ENVIRONMENT AND SESSION SET UP ####################################################
    ########################################################################################

    # set the environment variable
    os.environ["KERAS_BACKEND"] = "tensorflow"
    # Silence INFO logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    # Prevent usage of GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
    cfg = yaml.load(open("config.yml", "r"), Loader=yaml.Loader)

    # Assign variables
    dataset_name = cfg["dataset name"]
    dataset_path = cfg["data path"]
    splits_path = cfg["splits path"]
    number_of_splits = cfg["number of splits"]
    impute_data = cfg["impute data"]
    
    models_to_use,explainability_measures = config_handler(cfg)
            
    subsampling_types = cfg["subsampling to use"]
    performance_scores = cfg["final performance measures"]
    fixed_parameters = cfg["fixed hyperparameters"]
    models_folder = cfg["models folder path"]
    importance_folder = cfg["importance folder path"]


    ########################################################################################
    ###### GET TRAINING AND TEST DATA ######################################################
    ########################################################################################

    data = ClinicalDataset(name=dataset_name, path=dataset_path)

    # Load the training-test splits as a class instance variable using the
    # assign_train_test_sets function with only the path to the splits file
    data.assign_train_test_splits(splits_path)

    # Preprocess data
    if impute_data:
        data.impute(number_of_splits=number_of_splits, imputation_type="mean/mode")

    data.normalize(number_of_splits=number_of_splits)

    print("Number of patients in dataset: " + str(len(data.X)))


    ########################################################################################
    ###### CALCULATE WEIGHTS, SHAP AND DEEP TAYLOR DECOMPOSITION VALUES ####################
    ########################################################################################

    # Check if the weights folder path to save already exists. If not, create folder.
    if not os.path.exists(importance_folder):
        os.makedirs(importance_folder)

    # Iterate over subsampling types
    for subs in subsampling_types:
        data.subsample_training_sets(number_of_splits, subs)

        # Iterate over models
        for idx,mdl in enumerate(models_to_use):
            file_suffix = ".h5" if mdl == "MLP" else ".pkl"
            for expl in explainability_measures[idx]:
                path_to_imp_values = (f"{importance_folder}/{mdl}_{expl}_{subs}_subsampling.csv")

                # Check if the importance values file path already exists. If it exists, return
                # to previous loop.
                if not os.path.isfile(path_to_imp_values):
                    # Create temporary variable to store the calculated importance values from
                    # each split.
                    tmp_values = np.zeros((number_of_splits, len(data.cols) - 1))
    
                    # Iterate over splits.
                    for i in range(number_of_splits):
                        # Get main path of saved models.
                        path_to_model = f"{models_folder}/{mdl}_model_{subs}_subsampling_split_{i+1}{file_suffix}"
    
                        # Create model instance for the current split and load the model of
                        # the current split.
                        model = eval(mdl)(name=mdl, dataset=data.splits[i], fixed_params=fixed_parameters)
                        model.load_model(path_to_model)
    
                        # Get model weights for linear classifiers directly.
                        if mdl in ["GLM", "Lasso", "ElasticNet"]:
                            if expl == 'weights':
                                weights = ["{:.5f}".format(weight) for weight in model.best_model.coef_[0]]
                                tmp_values[i, :] = weights    
                            if expl == 'shap':
                                shap_values = linear_shap_value(model)
                                shaps = ["{:.5f}".format(shap) for shap in shap_values]
                                tmp_values[i, :] = shaps
                        # Calculate shap values for the non-linear Catboost classifier.
                        elif mdl == "Catboost":
                            if expl == 'shap':
                                shap_values = calc_shap_values(dataset=data, model=model)
                                shaps = ["{:.5f}".format(shap) for shap in shap_values]
                                tmp_values[i, :] = shaps
                        elif mdl == "MLP":
                            if expl == 'shap':
                                shap_values = calc_kernel_shap_values(model)
                                shaps = ["{:.5f}".format(shap) for shap in shap_values]
                                tmp_values[i, :] = shaps
                            elif expl == 'taylor':
                                dt_values = calc_deep_taylor_values(model=model)
                                tmp_values[i, :] = dt_values
                        elif mdl == "SVMC":
                            if expl == 'shap':
                                shap_values = calc_kernel_shap_values(model)
                                shaps = ["{:.5f}".format(shap) for shap in shap_values]
                                tmp_values[i, :] = shaps
                        elif mdl == "NB":
                            if expl == 'shap':
                                shap_values = calc_kernel_shap_values(model)
                                shaps = ["{:.5f}".format(shap) for shap in shap_values]
                                tmp_values[i, :] = shaps
                            
                            
                    df_values = pd.DataFrame(tmp_values, columns=list(data.X))
                    df_values.to_csv(path_to_imp_values)
    
                    print(f"Saved {mdl} feature importance values with {subs} subsampling.")
