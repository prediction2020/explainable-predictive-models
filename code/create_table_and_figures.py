import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import os
import itertools
from scipy.stats import iqr

import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from plotting_helper import *

########################################################################################
#### ENVIRONMENT AND SESSION SET UP ####################################################
########################################################################################

# set the environment variable
os.environ["KERAS_BACKEND"] = "tensorflow"
# Silence INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Prevent usage of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""


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

# Assign variables
dataset_name = cfg['dataset']['name']
models_trained = cfg['models to use']
subsampling_types = cfg['subsampling to use']
performance_measures = cfg['final performance measures']
scores_path = cfg['save options']['scores path']
weights_path = cfg['save options']['weights path']
save_figures = cfg['save options']['figures path']
splits_path = cfg['dataset']['splits path']
save_models = cfg['save options']['models path']
feature_dict = {'DG_AG':'age', 'DWI': 'DWI lesion volume', 'AD_NIH':'NIHSS', 'RF_HC':'hypercholesterolemia','AT_LY':'thrombolysis treatment',
                    'DG_SEX':'gender', 'CH':'cardiac history', 'RF_DM':'diabetes'}


figure = True
table = False


########################################################################################
###### CREATE TABLE OF ALL PERFORMANCE SCORES ##########################################
########################################################################################

if table == True:
    # Check if the scores file path already exists.
    if not os.path.isfile(scores_path+'/'+dataset_name+'_performance_results.csv'):
        values = ['median', 'iqr']
        iterables= [subsampling_types, models_trained, values]
        multiidx = pd.MultiIndex.from_product(iterables, names = ['Subsampling Type','Model','Value'])
        scoring = [' (training)',' (test)']
        performance_index = [''.join(item) for item in itertools.product([pm.replace('_',' ') for pm in performance_measures],scoring)]

        df = pd.DataFrame([],columns=performance_index,index = multiidx)

        for perf, subs, mdl in itertools.product(performance_measures , subsampling_types,models_trained):
            model_results = pd.read_csv(scores_path+ '/'+mdl+'_'+perf+'_scores_'+subs+'_subsampling.csv',index_col=0)
            df.loc[(subs,mdl),perf.replace('_',' ') +' (training)'] = "{:.2f}".format(np.median(model_results,axis=0)[0]),"{:.2f}".format(iqr(model_results,axis=0)[0])
            df.loc[(subs,mdl),perf.replace('_',' ') +' (test)']= "{:.2f}".format(np.median(model_results,axis=0)[1]),"{:.2f}".format(iqr(model_results,axis=0)[1])

        df.to_csv(scores_path+'/'+dataset_name+'_performance_results.csv')


########################################################################################
###### CREATE PLOTS OF PERFORMANCE AND FEATURE RATING ##################################
########################################################################################
plot_performances = False
plot_feature_imp = True

if figure ==True:

    # Check if the feature rating figures folder path to save already exists. If not, 
    # create folder.
    if not os.path.exists(save_figures+ '/feature_ratings'):
        os.makedirs(save_figures+ '/feature_ratings')

    # Check if the performance figures folder path to save already exists. If not, 
    # create folder.
    if not os.path.exists(save_figures+ '/final_performance_scores'):
        os.makedirs(save_figures+ '/final_performance_scores')

    # PLOT PERFORMANCE
    # Iterate over subsampling types
    if plot_performances == True:
        for subs in subsampling_types:
            all_scores = dict(zip(performance_measures, [None]*len(performance_measures)))
            for perf in performance_measures:
                scores = dict()
                #if not os.path.isfile(save_figures+ '/final_performance_scores/'+perf+'_scores_'+subs+'_subsampling.png'):
                for mdl in models_trained:           
                    tmp_score = {mdl: pd.read_csv(scores_path+ '/'+mdl+'_'+perf+'_scores_'+subs+'_subsampling.csv',index_col=0)}
                    scores.update(tmp_score)
                    #scores[mdl] = pickle.load(open(scores_path+ '/'+mdl+'_'+perf+'_scores_'+subs+'_subsampling.pkl','rb'))
                plot_performance(scores, models_trained, subs, perf, save_figures+ '/final_performance_scores')
                all_scores[perf] = scores

            plot_all_performance(all_scores,models_trained, subs, save_figures+ '/final_performance_scores')



    # PLOT FEATURE IMPORTANCE
    # Iterate over subsampling types
    if plot_feature_imp == True:
        for subs in subsampling_types:
            weights = dict()
            for mdl in models_trained:          
                if mdl not in ['Catboost','MLP']:
                    tmp_weights= {mdl: pd.read_csv(weights_path+ '/best_'+mdl+'_weights_'+subs+'_subsampling.csv',index_col=0)}
                    tmp_weights[mdl] = tmp_weights[mdl].rename(columns= feature_dict)
                    weights.update(tmp_weights)

                elif mdl == 'Catboost':
                    shaps= {mdl: pd.read_csv(weights_path+ '/best_'+mdl+'_shap_values_'+subs+'_subsampling.csv',index_col=0)}
                    shaps[mdl] = shaps[mdl].rename(columns= feature_dict)
                elif mdl == 'MLP':
                    dts = {mdl: pd.read_csv(weights_path+ '/best_'+mdl+'_dt_values_'+subs+'_subsampling.csv',index_col=0)}
                    dts[mdl] = dts[mdl].rename(columns= feature_dict)
                    #bps = {mdl: pd.read_csv(weights_path+ '/best_'+mdl+'_bp_values_'+subs+'_subsampling.csv',index_col=0)}
                    #bps[mdl] = bps[mdl].rename(columns= feature_dict)
                
        #    plot_linear_model_rating(weights,subs,save_figures+ '/feature_ratings')
        #    plot_catboost_rating(shaps, subs, save_figures+ '/feature_ratings')
            weights.update(shaps)
            weights.update(dts)
            #print(weights)
            plot_predictors_rating(weights, subs, save_figures+ '/feature_ratings')

    
    
