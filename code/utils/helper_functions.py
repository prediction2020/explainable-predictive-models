import numpy as np
import pandas as pd
import catboost as cat
import keras 
import matplotlib.pyplot as plt
from scipy.stats import iqr
import seaborn as sns
from keras import activations
import shap
from sklearn import preprocessing
from vis.utils import utils
import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.keras.graph as inkgraph
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score


def subsample(X,y,subsampling_type):
    df = pd.concat([X,y],axis=1)
    label = list(df)[-1]

    if subsampling_type=='random':
        df_bad = df[(df[label] == 1)]
        df_good = df[(df[label] == 0)]

        df_sub = pd.concat([df_good.sample(len(df_bad.index),random_state = 21),df_bad])
        X_new, y_new = df_sub.drop(label,axis=1), df_sub[[label]]

    elif subsampling_type=='none':
        X_new, y_new = X,y

    return X_new, y_new

def predict_probability(data,model,model_name):
        if model_name == 'MLP':
            output_activation = model.get_config()['layers'][-1]['config']['activation']
            if output_activation == 'softmax':
                probs = model.predict_proba(data)
            else:
                probs = model.predict(data)
        else:
            probs = model.predict_proba(data).T[1]

        return probs

def predict_class(data,model,model_name):
    preds = model.predict(data).astype('float64')

    if model_name == 'MLP':
        output_activation = model.get_config()['layers'][-1]['config']['activation']
        if output_activation == 'softmax':
            probs = model.predict(data)
            preds = probs.argmax(axis=-1)
        else:
            preds = model.predict_classes(data)

    return preds

def calc_perf_score(data,labels,model,model_name, score_name):
    if isinstance(labels, pd.DataFrame):
        labels.iloc[:,0] = labels.iloc[:,0].astype('float64')

    if score_name == 'AUC':
        probs = predict_probability(data, model, model_name)
        score = roc_auc_score(labels, probs)


    else:
        preds = predict_class(data, model, model_name)
        if score_name == 'accuracy':
            score = accuracy_score(labels,preds)
        elif score_name == 'f1':
            score = f1_score(labels,preds,pos_label=1)
        elif score_name == 'average_class_accuracy':
            recall = recall_score(labels,preds,average=None)
            score = 2/(1/recall[0]+1/recall[1])

    return score

def calc_shap_values(dataset,model):
    explainer = shap.TreeExplainer(model.best_model)
    cat_features = [list(model.X_tr).index(dataset.cat_preds[i]) for i in range(len(dataset.cat_preds))]
    shap_values = explainer.shap_values(cat.Pool(model.X_tr,
                                                 model.y_tr, 
                                                 cat_features = cat_features))
    shap_values_mean_over_samples = np.mean(shap_values,axis=0)

    return shap_values_mean_over_samples

def calc_deep_taylor_values(model):
    # Predict training and test probabilities
    test_probs = predict_probability(model.X_te, model.best_model, 'MLP')
    train_probs = predict_probability(model.X_tr, model.best_model, 'MLP')

    # Set last layer activation to linear. If this swapping is not done, the 
    # results might be suboptimal
    Model.best_model.layers[-1].activation = activations.linear
    stripped_model = utils.apply_modifications(model.best_model)

    # Calculate class weights
    train_input_weights = train_probs
    train_input_weights[np.where(model.y_tr == 0)] = 1-train_input_weights[np.where(model.y_tr == 0)]

    class_idx = 0 # if the activation of last layer was sigmoid
    last_layer_idx = utils.find_layer_idx(model.best_model, 'dense_2')

    # Calculate global gradients of all patients (deep taylor)
    seed_input = Model.X_tr.values
    # The deep taylor is bounded to a range which should be defined based on the input range:
    input_range = [min(seed_input.flatten()),max(seed_input.flatten())] 
    gradient_analyzer = innvestigate.create_analyzer("deep_taylor.bounded",        # analysis method identifier
                                                    stripped_model, # model without softmax output
                                                    low = input_range[0],
                                                    high =input_range[1])  

    # Some analyzers require training. You will get a warning message for the redundently fitted analyzer, you can ignore it
    gradient_analyzer.fit(seed_input, batch_size=16, verbose=1)
    analysis = gradient_analyzer.analyze(seed_input)
    # Calculate simple average:
    avg_analysis = np.expand_dims(np.mean(analysis,axis=0),axis = 0)

    # Calculate score based average
    t_analysis = np.transpose(analysis,(1,0))
    train_input_weights_s = np.squeeze(train_input_weights)
    score_avg_analysis = np.expand_dims(np.dot(t_analysis,train_input_weights_s),axis = 0)

    return score_avg_analysis

def plot_performance(scores, model_names,sub_type,path):
    perfs = list(scores.keys())
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

    model_names_ticks = [m.replace('Catboost', 'Tree Boosting').replace('ElasticNet','Elastic Net') for m in model_names]

    ax[0].set_ylabel('Value', size=16)
    plt.xticks(range(model_count),model_names_ticks)
    plt.suptitle('Final Performance Scores',size=20,y=1.04)
    plt.tight_layout()
    fig.savefig(f'{path}/performance_scores_{sub_type}_subsampling.png',bbox_inches='tight')
    plt.close(fig)

def plot_features_rating(values, sub_type, path):
    models = list(values.keys())
    sns.set(style = "white", rc={"xtick.labelsize": 10, "ytick.labelsize": 18})
    
    fig,ax = plt.subplots(1,len(models),sharey=True,sharex = True,figsize=(16,5))
    for i,mdl in enumerate(models):
        value = values[mdl].copy()
        value[:] = preprocessing.normalize(value)
        sns.barplot( data= abs(value),orient= 'h',ci='sd',errwidth=0.9,estimator = np.mean,ax = ax[i])
        
        if mdl == 'Catboost':
            ax[i].set_xlabel('|shap values|',size=16)
        elif mdl == 'MLP':
            ax[i].set_xlabel('|deep taylor values|',size=16)
        else:
            ax[i].set_xlabel('|weights|',size=16)
        ax[i].set_title(mdl.replace('Catboost','Tree Boosting').replace('Elasticnet', 'Elastic Net'),size=20)
    plt.suptitle('Clinical Features Importance Rating',size=20,y=1.02)
    
    #plt.show()
    fig.savefig(f'{path}/clinical_features_rating_{sub_type}_subsampling.png',bbox_inches='tight')#,transparent=True)
    plt.close(fig)