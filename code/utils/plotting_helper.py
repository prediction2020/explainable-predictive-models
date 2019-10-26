import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg') # for remote usage

import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from scipy.stats import iqr
import seaborn as sns


def plot_performance(score,model_names,sub_type,score_name,path):
    mean_tr, std_tr = [score[m].mean(axis=0)[0] for m in model_names], [score[m].std(axis=0)[0] for m in model_names]
    mean_te, std_te = [score[m].mean(axis=0)[1] for m in model_names], [score[m].std(axis=0)[1] for m in model_names]
    median_tr, iqr_tr = [np.median(score[m],axis=0)[0] for m in model_names], [iqr(score[m],axis=0)[0] for m in model_names]
    median_te, iqr_te = [np.median(score[m],axis=0)[1] for m in model_names] ,[iqr(score[m],axis=0)[1] for m in model_names]

    model_count = len(model_names)

    fig = plt.figure(figsize=(4,6))
    plt.errorbar(range(model_count),median_tr,yerr = iqr_tr,marker='o',markersize='10',ls='None',alpha=0.6,capsize=4,label= 'training '+score_name)
    plt.errorbar(range(model_count),median_te,yerr = iqr_te,marker='o',markersize='10',ls='None',alpha=0.6,capsize=4, label= 'test '+score_name)
    model_names_ticks = [m.replace('Catboost', 'Tree Boosting').replace('Elasticnet','Elastic Net') for m in model_names]
    plt.xticks(range(model_count),model_names_ticks)
    plt.legend(loc= 'upper left')
    plt.ylabel('Value')
    if score_name=='AUC':
        plt.ylim(0.5,1)
    else:
        plt.ylim(0,1)
    plt.title('%s scores (%s subsampling)'%(score_name.replace('_',' '),sub_type.replace('none','no')))
    plt.tight_layout()
    fig.savefig(path+'/'+score_name+'_scores_'+sub_type+'_subsampling.png',bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_all_performance(scores, model_names,sub_type,path):
    perfs = scores.keys()
    model_count = len(model_names)

    fig, ax = plt.subplots(1,len(perfs),sharey =True, sharex=True, figsize=(30,5))
    plt.style.use('seaborn-notebook')
    for i, perf in enumerate(perfs):
        score = scores[perf]
        mean_tr, std_tr = [score[m].mean(axis=0)[0] for m in model_names], [score[m].std(axis=0)[0] for m in model_names]
        mean_te, std_te = [score[m].mean(axis=0)[1] for m in model_names], [score[m].std(axis=0)[1] for m in model_names]
        median_tr, iqr_tr = [np.median(score[m],axis=0)[0] for m in model_names], [iqr(score[m],axis=0)[0] for m in model_names]
        median_te, iqr_te = [np.median(score[m],axis=0)[1] for m in model_names] ,[iqr(score[m],axis=0)[1] for m in model_names]      

        ax[i].errorbar(range(model_count),median_tr,yerr = iqr_tr,marker='o',markersize='12',ls='None',alpha=0.6,capsize=4,label= 'training '+perf.replace('average','avg').replace('_', ' ').replace('accuracy','acc'))
        ax[i].errorbar(range(model_count),median_te,yerr = iqr_te,marker='o',markersize='12',ls='None',alpha=0.6,capsize=4, label= 'test '+perf.replace('average','avg').replace('_', ' ').replace('accuracy','acc'))
        ax[i].legend(loc= 'lower left',fontsize=15)
        ax[i].set_title(perf.replace('_',' '), size=20)
        ax[i].set_ylim(0.3,1)
        ax[i].tick_params(axis='both',labelsize=13)
        #ax[i].tick_params(axis='y',labelsize=12)

    model_names_ticks = [m.replace('Catboost', 'Tree Boosting').replace('Elasticnet','Elastic Net') for m in model_names]

    ax[0].set_ylabel('Value', size=16)
    plt.xticks(range(model_count),model_names_ticks)
    plt.suptitle('Final Performance Scores',size=20,y=1.04)
    plt.tight_layout()
    fig.savefig(path+'/all_performance_scores_'+sub_type+'_subsampling.png',bbox_inches='tight')#,transparent=True)
    #plt.show()
    plt.close(fig)


def plot_linear_model_rating(weights, sub_type, path):
    models = weights.keys()
    fig,ax = plt.subplots(1,len(models),sharey=True,sharex = True,figsize=(15,6))
    for i,mdl in enumerate(models):
        sns.barplot( data= abs(weights[mdl]),orient= 'h',ax = ax[i])
        ax[i].set_xlabel('|weights|')
        ax[i].set_title(mdl)
    plt.suptitle('Clinical predictor ratings with linear models')
    #plt.show()
    fig.savefig(path+'/clinical_predictor_ratings_linear_models_'+sub_type+'_subsampling.png',bbox_inches='tight')
    plt.close(fig)

def plot_catboost_rating(shaps, sub_type, path):
    fig = plt.figure(figsize=(9,6))
    sns.barplot( data= abs(shaps['Catboost']),orient= 'h')
    plt.xlabel('|shap_values|')
    plt.title('Clinical predictor ratings with Catboost')
    #plt.show()
    fig.savefig(path+'/clinical_predictor_ratings_Catboost_model_'+sub_type+'_subsampling.png',bbox_inches='tight')
    plt.close(fig)

def plot_predictors_rating(values, sub_type, path):
    models = values.keys()
    sns.set(style = "white", rc={"xtick.labelsize": 10, "ytick.labelsize": 18})
    
    fig,ax = plt.subplots(1,len(models),sharey=True,sharex = True,figsize=(16,5))
    for i,mdl in enumerate(models):
        value = values[mdl]
        #print(value)
        value /= np.abs(value.values.max())
        #value = value.apply(lambda x: x/abs(x).max(), axis=0) # normalize over runs for each feature (column-wise operation)
        print(value)
        sns.barplot( data= abs(value),orient= 'h',ax = ax[i])
        
        if mdl == 'Catboost':
            ax[i].set_xlabel('|shap values|',size=16)
        elif mdl == 'MLP':
            ax[i].set_xlabel('|deep taylor values|',size=16)
        else:
            ax[i].set_xlabel('|weights|',size=16)
        ax[i].set_title(mdl.replace('Catboost','Tree Boosting').replace('Elasticnet', 'Elastic Net'),size=20)
    plt.suptitle('Clinical Predictors Importance Rating',size=20,y=1.02)
    
    #plt.show()
    fig.savefig(path+'/clinical_predictor_ratings_all_models_'+sub_type+'_subsampling.png',bbox_inches='tight')#,transparent=True)
    plt.close(fig)


def plot_evolution(loss,val_loss,auc,val_auc,test_auc,params,save_path):
    eps = range(1, len(loss)+1)
    fig, axs = plt.subplots(1,2, figsize= (20,8))
    axs[0].plot(eps, loss, 'b', label ='training loss')
    axs[0].plot(eps, val_loss, 'r', label ='validation loss')
    axs[0].set_title('Loss over epochs')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].legend()
    axs[1].plot(eps, auc, 'b', label ='training auc')
    if val_auc:
        axs[1].plot(eps, val_auc, 'r', label ='validation auc')
    axs[1].plot(eps, test_auc, 'g', label ='test auc')
    axs[1].set_title('AUC over epochs')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('AUC')
    axs[1].legend()
    fig.suptitle(str(params), fontsize=12, fontweight='bold')
    fig.savefig(save_path)
    plt.close(fig)