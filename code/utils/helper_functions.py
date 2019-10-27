import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score


def subsample(X,y,subsampling_type):
    df = pd.concat([X,y],axis=1)
    label = list(df)[-1]

    if subsampling_type=='random':
        df_bad = df[(df[label] == 1)]
        df_good = df[(df[label] == 0)]

        df_sub = pd.concat([df_good.sample(len(df_bad.index)),df_bad])
        X_new, y_new = df_sub.drop(label,axis=1), df_sub[[label]]

    elif subsampling_type=='none':
        X_new, y_new = X,y

    return X_new, y_new

def predict_probability(data,model,model_name):
        if model_name == 'MLP':
            if self.def_params['out_activation'] == 'softmax':
                probs = model.predict_proba(data,model_name)
            else:
                probs = model.predict(data,model_name)
        else:
            probs = model.predict_proba(data,model_name).T[1]

        return probs

def predict_class(data,model,model_name):
    preds = model.predict(data).astype('float64')

    if model_name == 'MLP':
        if self.def_params['out_activation'] == 'softmax':
            probs = model.predict(data)
            preds = probs.argmax(axis=-1)
        else:
            preds = model.predict_classes(data)

    return preds

def calculate_performance_score(data,labels,model,model_name, score_name):
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