import numpy as np
import pickle
import time
import os

import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import preprocessing

import catboost as cat

import keras
import tensorflow as tf
os.environ["KERAS_BACKEND"] = 'tensorflow'
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from helper import *
from keras_helper import *

from abc import abstractmethod


class clinical_dataset(dataset):
    def __init__(self,name):
        super().__init__(name)
        self.datatype = 'clinical'

    def load_data(self,datasource):
        if isinstance(datasource,str):
            self.df = pd.read_pickle(datasource)
        else:
            self.df = datasource
        self.cats = self.df.columns[self.df.dtypes=='category']
        self.cols = list(self.df)
        self.label = list(self.df)[-1]
        self.X, self.y = self.df.drop(self.label,axis=1), self.df[[self.label]]

    def preprocess(self):
        if 'AT_ST' in self.cols:
            self.df['AT_ST'] = self.df['AT_ST'] / np.timedelta64(1, 'h')
            while self.df['AT_ST'].max()>100:
                self.df = self.df.drop(self.df['AT_ST'].idxmax())

        for i, col in enumerate(self.cols):
            self.df[col] = self.df[col].astype('float64')

        # Centering numeric data
        self.num_data = list(set(self.cols) - set(self.cats))
        self.X.loc[:,self.num_data] = preprocessing.StandardScaler().fit_transform(self.X.loc[:,self.num_data])

        #self.df = pd.concat([self.X, self.y],axis=1)



class GLM_cv(model):
    def run(self,*args):
        self.best_model = LogisticRegression(**self.def_params)
        self.best_model.fit(self.X_tr,np.ravel(self.y_tr))



class Lasso_cv(model):
    def run(self,my_cv,cv_score):
        model = LogisticRegression(**self.def_params)

        gsearch = GridSearchCV(estimator= model, cv= my_cv, param_grid=self.tuning_params, scoring=cv_score,iid=True,n_jobs=-1)
        gsearch.fit(self.X_tr.values.astype('float'), np.ravel(self.y_tr.values.astype('float')))

        self.best_params = gsearch.best_params_
        self.best_model = LogisticRegression(**self.def_params)
        self.best_model.set_params(**self.best_params)
        self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class Elasticnet_cv(model):
    def run(self,my_cv,cv_score):
        model = SGDClassifier(**self.def_params)

        gsearch = GridSearchCV(estimator=model, cv=my_cv, param_grid=self.tuning_params, scoring=cv_score,iid=True,n_jobs=-1)
        gsearch.fit(self.X_tr.values.astype('float'), np.ravel(self.y_tr.values.astype('float')))

        self.best_params = gsearch.best_params_
        self.best_model = SGDClassifier(**self.def_params)
        self.best_model.set_params(**self.best_params)
        self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class Catboost_cv(model):
    def run(self,my_cv,cv_score):
        cats = self.X_tr.columns[self.X_tr.dtypes=='category']
        model_params = self.def_params.copy()

        best_AUC = 0.5
        for tune in ParameterGrid(self.tuning_params):
            model_params.update(tune)

            AUC_val = []
            for train, val in my_cv.split(self.X_tr,self.y_tr):
                X_train, y_train = self.X_tr.iloc[train], self.y_tr.iloc[train]
                X_val, y_val = self.X_tr.iloc[val], self.y_tr.iloc[val]
                train_pool = cat.Pool(X_train, y_train, cat_features= [list(self.X_tr).index(cats[i]) for i in range(len(cats))])
                validate_pool = cat.Pool(X_val, y_val, cat_features= [list(self.X_tr).index(cats[i]) for i in range(len(cats))])

                model = cat.CatBoostClassifier(**model_params)
                model.fit(train_pool,eval_set=validate_pool,logging_level='Silent')
                probs_val = model.predict_proba(X_val).T[1]
                AUC = roc_auc_score(np.array(y_val.astype('float')), probs_val)
                AUC_val.append(AUC)

            AUC_val = np.mean(AUC_val)

            if AUC_val > best_AUC:
                best_AUC = AUC_val
                self.best_model = model

        self.best_params = self.best_model.get_params()


class MLP_cv(model):
    def run(self,my_cv,cv_score):
        # Setting default parameters
        params = self.def_params.copy()

        # Fix seed
        np.random.seed(1)
        tf.set_random_seed(2)

        # Start Gridsearch
        best_AUC = 0.5
        for tune in ParameterGrid(self.tuning_params):
            params.update(tune)

            AUC_val = []
            for train, val in my_cv.split(self.X_tr,self.y_tr):
                X_train, y_train = self.X_tr.iloc[train], self.y_tr.iloc[train]
                X_val, y_val = self.X_tr.iloc[val], self.y_tr.iloc[val]

                e_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.005, patience = params['iter_patience'], mode='min')
                callbacks = [e_stop]
                optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'])

                model = Sequential()
                model.add(Dense(params['num_neurons'],input_dim = len(list(self.X_tr)), kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l1(params['l1_ratio'])))
                model.add(Dropout(params['dropout_rate']))
                model.add(Dense(1, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l1(params['l1_ratio'])))
                model.compile(loss = params['loss_func'], optimizer = optimizer)

                history = model.fit(X_train, y_train, callbacks= callbacks, validation_data = (X_val, y_val), epochs = params['epochs'], batch_size = params['batch_size'], verbose = 0)

                probs_val = model.predict_proba(X_val).T[0]
                AUC = roc_auc_score(y_val, probs_val)
                AUC_val.append(AUC)

            AUC_val = np.mean(AUC_val)

            if AUC_val > best_AUC:
                best_AUC = AUC_val
                self.best_params = tune

            keras.backend.clear_session()

        e_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.005, patience = params['iter_patience'], mode='min')
        callbacks = [e_stop]
        optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = self.best_params['learning_rate'])

        model = Sequential()
        model.add(Dense(self.best_params['num_neurons'],input_dim = len(list(self.X_tr)), kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l2(self.best_params['l1_ratio'])))
        model.add(Dropout(self.best_params['dropout_rate']))
        model.add(Dense(1, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l2(self.best_params['l1_ratio'])))
        model.compile(loss = params['loss_func'], optimizer = optimizer)

        history = model.fit(self.X_tr, self.y_tr, callbacks= callbacks, validation_data = (self.X_te, self.y_te), epochs = params['epochs'], batch_size = self.best_params['batch_size'], verbose = 0)

        self.best_model = model



class MLP(model):
    def run(self,path_to_save_model_on_inner_tr = None):
        # Setting default parameters
        params = self.def_params.copy()
        num_features = self.X_tr.shape[1]

        # Fix seed
        np.random.seed(1)
        tf.set_random_seed(2)


        # Start Gridsearch
        best_AUC = 0.5

        i = 0


        for tune in ParameterGrid(self.tuning_params):
            params.update(tune)

            e_stop = EarlyStopping(monitor = 'val_loss', min_delta = params['min_delta'], patience = params['iter_patience'], mode='min')
            ep_val_inner = EpochEvaluation(validation_data = (self.X_val, self.y_val),training_data = (self.X_tr, self.y_tr), test_data=(self.X_te,self.y_te),batch_size = 128)

            callbacks = [e_stop, ep_val_inner]
            optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'], decay =params['lr_decay'] )

            model = Sequential()
            model.add(Dense(params['num_neurons'],input_dim = num_features, kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
            model.add(Dropout(params['dropout_rate']))
            if params['num_hidden_layers']==2:
                model.add(Dense(np.int(np.floor(params['num_neurons']/2)),kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
                model.add(Dropout(params['dropout_rate']))
            if params['out_activation'] == 'softmax':
                model.add(Dense(2, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
            else:
                model.add(Dense(1, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
            model.compile(loss = params['loss_func'], optimizer = optimizer)

            history_inner = model.fit(self.X_tr, self.y_tr, callbacks= callbacks, validation_data = (self.X_val, self.y_val), epochs = params['epochs'], batch_size = params['batch_size'], verbose = 0)

            AUC_v = ep_val_inner.val_roc_auc[-1]

            i +=1
            if i%10 == 0:
                print(i)

            if AUC_v > best_AUC:
                best_AUC = AUC_v
                self.best_params = tune
                if isinstance(path_to_save_model_on_inner_tr, str):
                    model.save(path_to_save_model_on_inner_tr)

                self.loss_tr_inner = history_inner.history['loss']
                self.loss_val_inner = history_inner.history['val_loss']
                self.AUC_tr_inner = ep_val_inner.roc_auc
                self.AUC_val_inner = ep_val_inner.val_roc_auc
                self.AUC_te_inner = ep_val_inner.test_roc_auc

            
            keras.backend.clear_session()

        params.update(self.best_params)
        print(params)
        

        e_stop_last = EarlyStopping(monitor = 'val_loss', min_delta = params['min_delta'], patience = params['iter_patience'], mode='min')
        ep_val_outer = EpochEvaluation(validation_data = (self.X_val, self.y_val),training_data = (self.X_train, self.y_train), test_data=(self.X_te,self.y_te),batch_size = 128)

        callbacks = [e_stop_last, ep_val_outer]
        optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'], decay =params['lr_decay'] )

        model = Sequential()
        model.add(Dense(params['num_neurons'],input_dim = num_features, kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
        model.add(Dropout(params['dropout_rate']))
        if params['num_hidden_layers']==2:
            model.add(Dense(np.int(np.floor(params['num_neurons']/2)),kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
            model.add(Dropout(params['dropout_rate']))
        if params['out_activation'] == 'softmax':
            model.add(Dense(2, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
        else:
            model.add(Dense(1, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
        model.compile(loss = params['loss_func'], optimizer = optimizer)

        history_outer = model.fit(self.X_train, self.y_train, callbacks= callbacks, validation_data = (self.X_te, self.y_te), epochs = params['epochs'], batch_size = params['batch_size'], verbose = 0)

        self.loss_tr_outer = history_outer.history['loss']
        self.loss_val_outer = history_outer.history['val_loss']
        self.AUC_tr_outer = ep_val_outer.roc_auc
        self.AUC_te_outer = ep_val_outer.test_roc_auc

        self.best_model = model
