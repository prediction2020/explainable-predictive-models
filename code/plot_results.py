"""
File name: plot_results.py
Author: Esra Zihni
Date created: 27.07.2018



"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import os
import itertools
from scipy.stats import iqr

import pandas as pd

from utils.helper_functions import plot_performance, plot_features_rating

########################################################################################
###### ASSIGN CONFIGURATION VARIABLES ##################################################
########################################################################################

# You need to add constructor in order to be able to use the join command in the yaml 
# file
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join(str(i) for i in seq)

yaml.add_constructor('!join',join)

# Read the config file
cfg = yaml.load(open('config.yml', 'r'),Loader=yaml.Loader)

# Assign variables to use
models_to_use = cfg['models to use']
subsampling_types = cfg['subsampling to use']
performance_measures = cfg['final performance measures']
scores_folder = cfg['scores folder path']
importance_folder = cfg['importance folder path']
figures_folder = cfg['figures folder path']
feature_dict = cfg['features']

########################################################################################
###### CREATE PLOTS OF PERFORMANCE AND FEATURE RATING ##################################
########################################################################################

# Check if the feature rating figures folder path to save already exists. If not, 
# create folder.
if not os.path.exists(f'{figures_folder}/feature_ratings'):
    os.makedirs(f'{figures_folder}/feature_ratings')

# Check if the performance figures folder path to save already exists. If not, 
# create folder.
if not os.path.exists(f'{figures_folder}/final_performance_scores'):
    os.makedirs(f'{figures_folder}/final_performance_scores')

# PLOT PERFORMANCE
# Iterate over subsampling types
for subs in subsampling_types:
    all_scores = dict(zip(performance_measures, [None]*len(performance_measures)))
    for perf in performance_measures:
        scores = dict()
        for mdl in models_to_use:           
            tmp_score = {mdl: pd.read_csv(f'{scores_folder}/{mdl}_{perf}_scores_{subs}_subsampling.csv',index_col=0)}
            scores.update(tmp_score)

        all_scores[perf] = scores

    plot_performance(scores = all_scores,
    				 model_names =models_to_use, 
    				 sub_type = subs, 
    				 path = f'{figures_folder}/final_performance_scores')



# PLOT FEATURE IMPORTANCE
# Iterate over subsampling types
for subs in subsampling_types:
    values = dict()
    for mdl in models_to_use:          
        if mdl not in ['Catboost','MLP']:
            tmp_weights= {mdl: pd.read_csv(f'{importance_folder}/{mdl}_weights_{subs}_subsampling.csv',index_col=0)}
            tmp_weights[mdl] = tmp_weights[mdl].rename(columns= feature_dict)
            values.update(tmp_weights)

        elif mdl == 'Catboost':
            shaps= {mdl: pd.read_csv(f'{importance_folder}/{mdl}_shap_values_{subs}_subsampling.csv',index_col=0)}
            shaps[mdl] = shaps[mdl].rename(columns= feature_dict)
            values.update(shaps)

        elif mdl == 'MLP':
            dts = {mdl: pd.read_csv(f'{importance_folder}/{mdl}_dt_values_{subs}_subsampling.csv',index_col=0)}
            dts[mdl] = dts[mdl].rename(columns= feature_dict)
            values.update(dts)

    plot_features_rating(values = values, 
    					 sub_type = subs, 
    					 path = f'{figures_folder}/feature_ratings')

