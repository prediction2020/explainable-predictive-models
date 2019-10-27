import numpy as np
import pickle
import time
import os
import pandas as pd
import catboost as cat
import keras
import tensorflow as tf
os.environ["KERAS_BACKEND"] = 'tensorflow'
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from abc import abstractmethod
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import preprocessing
from helper_funstions import calculate_performance_score

class model:
    def __init__(self,name,dataset,fixed_params=None,tuning_params=None):
        self.name = name
        self.X_tr = dataset['train_data']
        self.y_tr = dataset['train_labels']
        self.X_te = dataset['test_data']
        self.y_te = dataset['test_labels']

        self.fixed_params = fixed_params
        self.tuning_params = tuning_params

        if self.fixed_params.get('out_activation') is 'softmax':
            self.y_tr = pd.get_dummies(self.y_tr)
            self.y_te = pd.get_dummies(self.y_te)

    def load_model(self, path):

    @abstractmethod
    def run_gridsearch(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def evaluate_performance(self, score_name):
    	training_performance = calculate_performance_score(data = self.X_tr, labels = self.y_tr, model = self.best_model, model_name = self.name, score_name = score_name)
    	test_performance = calculate_performance_score(data = self.X_te, labels = self.y_te, model = self.best_model, model_name = self.name, score_name = score_name)
    	return training_performance, test_performance


class GLM(model):
    def train(self):
        self.best_model = LogisticRegression(**self.fixed_params)
        self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class Lasso(model):
    def run_gridsearch(self,cv,cv_score):
        model = LogisticRegression(**self.fixed_params)

        gsearch = GridSearchCV(estimator= model, cv= cv, param_grid=self.tuning_params, scoring=cv_score,iid=True,n_jobs=-1)
        gsearch.fit(self.X_tr.values.astype('float'), np.ravel(self.y_tr.values.astype('float')))

        self.best_tuning_params = gsearch.best_params_

    def train(self, use_gridsearch_results=True):
    	params = self.fixed_params.copy()
    	if use_gridsearch_results ==True:
    		params.update(self.best_tuning_params)
        self.best_model = LogisticRegression(**params)
        self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class ElasticNet(model):
    def run_gridsearch(self,cv,cv_score):
        model = SGDClassifier(**self.fixed_params)

        gsearch = GridSearchCV(estimator=model, cv=cv, param_grid=self.tuning_params, scoring=cv_score,iid=True,n_jobs=-1)
        gsearch.fit(self.X_tr.values.astype('float'), np.ravel(self.y_tr.values.astype('float')))

        self.best_tuning_params = gsearch.best_params_

    def train(self, use_gridsearch_results = True):
    	params = self.fixed_params.copy()
    	if use_gridsearch_results == True:
    		params.update(self.best_tuning_params)
        self.best_model = SGDClassifier(**params)
        self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class Catboost(model):
    def run_gridsearch(self,cv,cv_score):
        cats = self.X_tr.columns[self.X_tr.dtypes=='category']
        params = self.fixed_params.copy()

        best_AUC = 0.5
        for tune in ParameterGrid(self.tuning_params):
            params.update(tune)

            AUC_val = []
            for train, val in cv.split(self.X_tr,self.y_tr):
                X_train, y_train = self.X_tr.iloc[train], self.y_tr.iloc[train]
                X_val, y_val = self.X_tr.iloc[val], self.y_tr.iloc[val]
                train_pool = cat.Pool(X_train, y_train, cat_features= [list(self.X_tr).index(cats[i]) for i in range(len(cats))])
                validate_pool = cat.Pool(X_val, y_val, cat_features= [list(self.X_tr).index(cats[i]) for i in range(len(cats))])

                model = cat.CatBoostClassifier(**params)
                model.fit(train_pool,eval_set=validate_pool,logging_level='Silent')

                validation_AUC = calculate_performance_score(data = X_val, labels = np.array(y_val.astype('float')), model = model, model_name= self.name,score_name = cv_score)
                AUC_val.append(validation_AUC)

            AUC_val = np.mean(AUC_val)

            if AUC_val > best_AUC:
                best_AUC = AUC_val
                self.best_tuning_params = tune
                
    def train(self, use_gridsearch_results=True):
    	params = self.fixed_params.copy()
    	if use_gridsearch_results == True:
    		params.update(self.best_tuning_params)

    	train_pool = cat.Pool(self.X_tr, self.y_tr, cat_features= [list(self.X_tr).index(cats[i]) for i in range(len(cats))])
    	test_pool = cat.Pool(self.X_te, self.y_te, cat_features= [list(self.X_te).index(cats[i]) for i in range(len(cats))])

        self.best_model = cat.CatBoostClassifier(**params)
        self.best_model.fit(train_pool,eval_set= test_pool,logging_level='Silent')


class MLP(model):
    def run_gridsearch(self,cv,cv_score):
        # Setting fixed parameters
        params = self.fixed_params.copy()

        # Fix seed
        np.random.seed(1)
        tf.set_random_seed(2)

        # Start Gridsearch
        best_AUC = 0.5
        for tune in ParameterGrid(self.tuning_params):
            params.update(tune)

            AUC_val = []
            for train, val in cv.split(self.X_tr,self.y_tr):
                X_train, y_train = self.X_tr.iloc[train], self.y_tr.iloc[train]
                X_val, y_val = self.X_tr.iloc[val], self.y_tr.iloc[val]

                e_stop = EarlyStopping(monitor = params['monitor'], min_delta = params['min_delta'], patience = params['iter_patience'], mode=params['mode'])
                callbacks = [e_stop]
                optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'])

                model = Sequential()
                model.add(Dense(params['num_neurons'],input_dim = len(list(self.X_tr)), kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l1(params['l1_ratio'])))
                model.add(Dropout(params['dropout_rate']))
                model.add(Dense(1, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l1(params['l1_ratio'])))
                model.compile(loss = params['loss_func'], optimizer = optimizer)

                history = model.fit(X_train, y_train, callbacks= callbacks, validation_data = (X_val, y_val), epochs = params['epochs'], batch_size = params['batch_size'], verbose = 0)

                validation_AUC = calculate_performance_score(data = X_val, labels = np.array(y_val.astype('float')), model = model, model_name= self.name,score_name = cv_score)
                AUC_val.append(validation_AUC)

            AUC_val = np.mean(AUC_val)

            if AUC_val > best_AUC:
                best_AUC = AUC_val
                self.best_tuning_params = tune

            keras.backend.clear_session()

    def train(self, use_gridsearch_results = True):
    	params = self.fixed_params.copy()
    	if use_gridsearch_results == True:
    		params.update(self.best_tuning_params)

        e_stop = EarlyStopping(monitor = params['monitor'], min_delta = params['min_delta'], patience = params['iter_patience'], mode=params['mode'])
        callbacks = [e_stop]
        optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'])

        model = Sequential()
        model.add(Dense(params['num_neurons'],input_dim = len(list(self.X_tr)), kernel_initializer = params['weight_init'], activation = params['hidden_activation'], kernel_regularizer = keras.regularizers.l2(params['l1_ratio'])))
        model.add(Dropout(params['dropout_rate']))
        model.add(Dense(1, kernel_initializer = params['weight_init'], activation = params['out_activation'], kernel_regularizer = keras.regularizers.l2(params['l1_ratio'])))
        model.compile(loss = params['loss_func'], optimizer = optimizer)

        history = model.fit(self.X_tr, self.y_tr, callbacks= callbacks, validation_data = (self.X_te, self.y_te), epochs = params['epochs'], batch_size = params['batch_size'], verbose = 0)

        self.best_model = model