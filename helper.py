import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

from abc import ABC,abstractmethod
from subsampling import *


class dataset:
    def __init__(self, name):
        self.name = name

    def assign_train_test_sets(self, path, test_size=None, runs=None):
        try:
            self.sets = pickle.load(open(path,'rb'))
            print('Loaded pre-existing %s training and test sets'%(self.datatype))

        except IOError:
            self.sets = []
            for i in range(runs):
                # Creating the same test data split to use in all models
                tmp_set = dict.fromkeys(['train_data','train_labels','test_data','test_labels'])
                self.X_tr,self.X_te,self.y_tr,self.y_te = train_test_split(self.X, self.y, test_size = test_size,stratify=self.y)

                # Save the sets to be used in all models
                tmp_set['train_data'] = self.X_tr
                tmp_set['train_labels'] = self.y_tr
                tmp_set['test_data'] = self.X_te
                tmp_set['test_labels'] = self.y_te

                self.sets.append(tmp_set)

            folder_to_save = path.split('/')[0]
            pickle.dump(self.sets, open(folder_to_save+'/'+self.name+'_'+self.datatype+'_train_test_sets.pkl', 'wb'))

        return self.sets

    def assign_train_val_test_sets(self,path,test_size=None,val_size=None,runs=None):
        try:
            self.sets = pickle.load(open(path,'rb'))
            print('Loaded pre-existing %s training, validation and test sets'%(self.datatype))

        except IOError:
            # Random state variable assures the split is the same every time the function is called
            self.sets = []
            #self.sets = dict.fromkeys(['train_data','train_labels','val_data','val_labels','test_data','test_labels'])
            for i in range(runs):
                # Creating the same test data split to use in all models
                tmp_set = dict.fromkeys(['train_data','train_labels','val_data','val_labels','test_data','test_labels'])

                self.X_train,self.X_te,self.y_train,self.y_te = train_test_split(self.X, self.y, test_size = test_size,stratify=self.y, random_state = 42+i)
                self.X_tr, self.X_val , self.y_tr, self.y_val = train_test_split(self.X_train, self.y_train, test_size = val_size, stratify= self.y_train, random_state=21+i)

                tmp_set['train_data'] = self.X_tr
                tmp_set['train_labels'] = self.y_tr
                tmp_set['val_data'] = self.X_val
                tmp_set['val_labels'] = self.y_val
                tmp_set['test_data'] = self.X_te
                tmp_set['test_labels'] = self.y_te

                self.sets.append(tmp_set)

            folder_to_save = path.split('/')[0]
            pickle.dump(self.sets, open(folder_to_save+'/'+self.name+'_'+self.datatype+'_train_val_test_set.pkl', 'wb'))

        return self.sets

    def subsample_training_sets(self,number_of_runs,sub_type):
        sets_to_use = []
        for i in range(number_of_runs):
            tmp_set = dict.fromkeys(['train_data','train_labels','test_data','test_labels'])
            tmp_set['test_data'] = self.sets[i]['test_data']
            tmp_set['test_labels'] = self.sets[i]['test_labels']
            #set_to_use = self.sets[i]
            tmp_set['train_data'], tmp_set['train_labels'] = subsample(self.sets[i]['train_data'], self.sets[i]['train_labels'],sub_type)
            sets_to_use.append(tmp_set)
        return sets_to_use


class model:
    def __init__(self,name,sets,default_params=None,tuning_params=None):
        self.name = name
        self.sets = sets
        self.X_tr = sets['train_data']
        self.y_tr = sets['train_labels']
        self.X_te = sets['test_data']
        self.y_te = sets['test_labels']

        self.def_params = default_params
        self.tuning_params = tuning_params

        if self.def_params.get('out_activation') is 'softmax':
            self.y_tr = pd.get_dummies(self.y_tr)
            self.y_te = pd.get_dummies(self.y_te)

        #if 'out_activation' in self.def_params.keys():
        #    if self.def_params['out_activation'] == 'softmax':
        #        self.y_tr = pd.get_dummies(self.y_tr)
        #        self.y_te = pd.get_dummies(self.y_te)

        if 'val_data' in list(sets):
            self.X_val = sets['val_data']
            self.y_val = sets['val_labels']
            if self.def_params['out_activation'] == 'softmax':
                self.y_val = pd.get_dummies(self.y_val)

            self.X_train = np.concatenate((self.X_tr,self.X_val))
            self.y_train = np.concatenate((self.y_tr,self.y_val))


    @abstractmethod
    def run(self):
        pass

    def predict_probability(self, data):
        if self.name == 'MLP' or self.name == 'MLP_multimodal':
            if self.def_params['out_activation'] == 'softmax':
                probs = self.best_model.predict_proba(data)
            else:
                probs = self.best_model.predict(data)
        else:
            probs = self.best_model.predict_proba(data).T[1]

        return probs

    def predict_class(self, data):
        preds = self.best_model.predict(data).astype('float64')

        if self.name == 'MLP' or self.name == 'MLP_multimodal':
            if self.def_params['out_activation'] == 'softmax':
                probs = self.best_model.predict(data)
                preds = probs.argmax(axis=-1)
            else:
                preds = self.best_model.predict_classes(data)

        return preds

    def get_performance_score(self, data, labels, score_name = 'AUC'):
        if isinstance(labels, pd.DataFrame):
            labels.iloc[:,0] = labels.iloc[:,0].astype('float64')

        if score_name == 'AUC':
            probs = self.predict_probability(data)
            score = roc_auc_score(labels, probs)


        else:
            preds = self.predict_class(data)
            if score_name == 'accuracy':
                score = accuracy_score(labels,preds)
            elif score_name == 'f1':
                score = f1_score(labels,preds,pos_label=1)
            elif score_name == 'average_class_accuracy':
                recall = recall_score(labels,preds,average=None)
                score = 2/(1/recall[0]+1/recall[1])

        return score
