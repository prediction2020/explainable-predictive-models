"""
File name: evaluate_models_performance.py
Author: Esra Zihni
Date created: 21.05.2018

This is the main script to evaluate models performance. It reads the implementation 
and path options from the config.yml script. It loads the training and test sets, 
loads the specified trained model(s) and calculates the training and test performance 
scores using the specified performance measure(s). It repeats the same process for 
the specified number of splits. Finally, it saves the calculated performance scores 
as csv files.
"""

import numpy as np
import yaml
import os
import keras
import pandas as pd

from utils.models import *
from utils.dataset import *

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

# Assign variables to use
dataset_name = cfg["dataset name"]
dataset_path = cfg["data path"]
splits_path = cfg["splits path"]
number_of_splits = cfg["number of splits"]
models_to_use = cfg["models to use"]
subsampling_types = cfg["subsampling to use"]
fixed_parameters = cfg["fixed hyperparameters"]
performance_scores = cfg["final performance measures"]
models_folder = cfg["models folder path"]
scores_folder = cfg["scores folder path"]


########################################################################################
###### GET TRAINING AND TEST DATA ######################################################
########################################################################################

# Load dataset
data = ClinicalDataset(name=dataset_name, path=dataset_path)

# Load the training-test splits as a class instance variable using the
# assign_train_test_sets function with only the path to the splits file
data.assign_train_test_splits(path=splits_path)

print("Number of patients in dataset: " + str(len(data.X)))


#########################################################################################
###### EVALUATE MODELS  AND SAVE PERFORMANCE SCORES #####################################
#########################################################################################

# Check if the scores folder path to save already exists. If not, create folder.
if not os.path.exists(scores_folder):
    os.makedirs(scores_folder)

# Iterate over subsampling types
for subs in subsampling_types:
    data.subsample_training_set(
        number_of_splits=number_of_splits, subsampling_type=subs
    )

    # Iterate over models.
    for mdl in models_to_use:
        file_suffix = ".h5" if mdl == "MLP" else ".pkl"

        # Iterate over performance scores.
        for scr in performance_scores:
            path_to_scores = (
                f"{scores_folder}/{mdl}_{scr}_scores_{subs}_subsampling.csv"
            )

            # Check if the scores file path already exists. If it exists, return to
            # previous loop.
            if not os.path.isfile(path_to_scores):
                # Create temporary variable to store the calculated scores from each split.
                tmp_scores = np.zeros((number_of_splits, 2))

                # Iterate over splits.
                for i in range(number_of_splits):
                    # Get main path of saved models.
                    path_to_model = f"{models_folder}/{mdl}_model_{subs}_subsampling_split_{i+1}{file_suffix}"

                    # Create model instance for the current split and load the model of
                    # the current split.
                    model = eval(mdl)(
                        name=mdl, dataset=data.splits[i], fixed_params=fixed_parameters
                    )

                    model.load_model(path=path_to_model)

                    # Calculate the performance score on the current training and test
                    # data.
                    score_tr, score_te = model.evaluate_performance(score_name=scr)

                    # Dump training and test scores of the current split to the
                    # temporary variable.
                    tmp_scores[i, 0] = score_tr
                    tmp_scores[i, 1] = score_te

                    # Clear keras session to prevent session from slowing down due to
                    # loaded models.
                    keras.backend.clear_session()

                # Store scores to dataframe and save.
                df_scores = pd.DataFrame(tmp_scores, columns=["training", "test"])
                df_scores.to_csv(path_to_scores)

                print(f"Saved {scr} scores for {mdl} model with {subs} subsampling.")
