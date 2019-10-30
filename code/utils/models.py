"""
File name: models.py
Author: Esra Zihni
Date created: 21.05.2018


"""

import numpy as np
import os
import pandas as pd
import pickle
import catboost as cat
import keras
import tensorflow as tf
os.environ["KERAS_BACKEND"] = 'tensorflow'
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression, SGDClassifier
from utils.helper_functions import calc_perf_score
from typing import List, Dict, Union

class Model(metaclass = ABCMeta):
	"""
	A metaclass used to represent any model.

	:param name: Name of the model 
	:param dataset: The dataset for the model to train on
	:param fixed_params: Hyperparameters that won't be used in model tuning
	:param tuning_params: Hyperparameters that can be used in model tuning

	.. py:meth:: Model.load_model(path)
		:param path: Path to model file.
	.. py:meth:: Model.run_gridsearch()	
	.. py_meth:: Model.train()
	.. py.meth:: Model.save_model(path)
		:param path: Path to model file.
	.. py.meth:: Model.evaluate_performance(score_name)
		:param score_name: Name of the performance measure.
		:return: Training and test performance scores 

	"""
	def __init__(self,name:str,dataset:Dict,fixed_params:Dict[str,Union[str,float]],tuning_params:Dict[str,Union[str,float]]=None):
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

	def load_model(self, path:str)-> None:
		"""
		Loads trained model from given path.
		:param path: Path to model file.
		"""
		if self.name =='MLP':
			self.best_model = load_model(path)
		else:
		  	self.best_model = pickle.load(open(path,'rb'))

	@abstractmethod
	def run_gridsearch(self)-> None: pass

	@abstractmethod
	def train(self)-> None: pass

	def save_model(self,path:str)-> None:
		"""
		Saves trained model to given path.
		:param path: Path to model file.
		"""
		if self.name =='MLP':
			self.best_model.save(path)
		else:
			pickle.dump(self.best_model, open(path,'wb'))


	def evaluate_performance(self, score_name:str)-> List[float]:
		"""
		Evaluates final model performance for the given performanc measure.
		:param score_name: Name of the performance measure.
		:return: Training and test performance scores 

		"""
		training_performance = calc_perf_score(data = self.X_tr, 
											   labels = self.y_tr, 
											   model = self.best_model, 
											   model_name = self.name, 
											   score_name = score_name)
		test_performance = calc_perf_score(data = self.X_te, 
										   labels = self.y_te, 
										   model = self.best_model, 
										   model_name = self.name, 
										   score_name = score_name)
		return training_performance, test_performance


class GLM(Model):
	"""
	A subclass of the Model metaclass used to represent a Generalized
	Linear Model (GLM).

	.. py:meth:: GLM.train()

	"""
	def train(self, **args):
		"""
		Trains a Generalized Linear Model (GLM).
		"""
		self.best_model = LogisticRegression(**self.fixed_params)
		self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class Lasso(Model):
		"""
		A subclass of the Model metaclass used to represent a Lasso model.

		.. py:meth:: Lasso.run_gridsearch()
		.. py:meth:: Lasso.train()

		"""
	def run_gridsearch(self,cv,cv_score:str)-> None:
		"""
		Performs a gridsearch over the tuning hyperparameters. Determines the 
		best hyperparameters based on the average validation performance 
		calculated over cross-validation folds.
		:param cv: A cross-validarion generator that determines the 
		cross-validation strategy.
		:param cv_score: Measure to evaluate predictions on the validation set. 
		"""
		model = LogisticRegression(**self.fixed_params)

		gsearch = GridSearchCV(estimator= model, 
							   cv= cv, 
							   param_grid=self.tuning_params, 
							   scoring=cv_score.replace('AUC','roc_auc'),
							   iid=True,
							   n_jobs=-1)
		gsearch.fit(self.X_tr.values.astype('float'), 
					np.ravel(self.y_tr.values.astype('float')))

		self.best_tuning_params = gsearch.best_params_

	def train(self, use_gridsearch_results:bool =True)-> None:
		"""
		Trains a Lasso model. 
		:param use_gridsearch_results: Determines whether to use 
		hyperparameters selected through gridsearch
		"""
		params = self.fixed_params.copy()
		if use_gridsearch_results == True:
			self.tuning_params = self.best_tuning_params
			
		params.update(self.tuning_params)

		self.best_model = LogisticRegression(**params)
		self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class ElasticNet(Model):
	def run_gridsearch(self,cv,cv_score:str)-> None:
		"""
		Performs a gridsearch over the tuning hyperparameters. Determines the 
		best hyperparameters based on the average validation performance 
		calculated over cross-validation folds.
		:param cv: A cross-validarion generator that determines the 
		cross-validation strategy.
		:param cv_score: Measure to evaluate predictions on the validation set. 
		"""
		model = SGDClassifier(**self.fixed_params)

		gsearch = GridSearchCV(estimator=model, 
							   cv=cv, 
							   param_grid=self.tuning_params, 
							   scoring=cv_score.replace('AUC','roc_auc'),
							   iid=True,
							   n_jobs=-1)
		gsearch.fit(self.X_tr.values.astype('float'), 
					np.ravel(self.y_tr.values.astype('float')))

		self.best_tuning_params = gsearch.best_params_

	def train(self, use_gridsearch_results:bool = True)-> None:
		"""
		Trains an Elastic Net model. 
		:param use_gridsearch_results: Determines whether to use 
		hyperparameters selected through gridsearch
		"""
		params = self.fixed_params.copy()
		if use_gridsearch_results == True:
			self.tuning_params = self.best_tuning_params
			
		params.update(self.tuning_params)

		self.best_model = SGDClassifier(**params)
		self.best_model.fit(self.X_tr,np.ravel(self.y_tr))


class Catboost(Model):
	def run_gridsearch(self,cv,cv_score:str)-> None:
		"""
		Performs a gridsearch over the tuning hyperparameters. Determines the 
		best hyperparameters based on the average validation performance 
		calculated over cross-validation folds.
		:param cv: A cross-validarion generator that determines the 
		cross-validation strategy.
		:param cv_score: Measure to evaluate predictions on the validation set. 
		"""
		cats = self.X_tr.columns[self.X_tr.dtypes=='category']
		cat_features = [list(self.X_tr).index(cats[i]) for i in range(len(cats))]
		params = self.fixed_params.copy()

		best_AUC = 0.5
		for tune in ParameterGrid(self.tuning_params):
			params.update(tune)

			AUC_val = []
			for train, val in cv.split(self.X_tr,self.y_tr):
				X_train, y_train = self.X_tr.iloc[train], self.y_tr.iloc[train]
				X_val, y_val = self.X_tr.iloc[val], self.y_tr.iloc[val]
				train_pool = cat.Pool(X_train, 
									  y_train, 
									  cat_features= cat_features)
				validate_pool = cat.Pool(X_val, 
										 y_val, 
										 cat_features= cat_features)

				model = cat.CatBoostClassifier(**params)
				model.fit(train_pool,
						  eval_set=validate_pool,
						  logging_level='Silent')

				validation_AUC = calc_perf_score(data = X_val, 
												 labels = np.array(y_val.astype('float')), 
												 model = model, 
												 model_name= self.name,
												 score_name = cv_score)
				AUC_val.append(validation_AUC)

			AUC_val = np.mean(AUC_val)

			if AUC_val > best_AUC:
				best_AUC = AUC_val
				self.best_tuning_params = tune

	def train(self, use_gridsearch_results:bool=True)-> None:
		"""
		Trains a Tree Boosting model. 
		:param use_gridsearch_results: Determines whether to use 
		hyperparameters selected through gridsearch
		"""
		params = self.fixed_params.copy()
		cats = self.X_tr.columns[self.X_tr.dtypes=='category']
		cat_features = [list(self.X_tr).index(cats[i]) for i in range(len(cats))]

		if use_gridsearch_results == True:
			self.tuning_params = self.best_tuning_params
			
		params.update(self.tuning_params)

		train_pool = cat.Pool(self.X_tr, 
							  self.y_tr, 
							  cat_features= cat_features)
		test_pool = cat.Pool(self.X_te, 
							 self.y_te, 
							 cat_features= cat_features)

		self.best_model = cat.CatBoostClassifier(**params)
		self.best_model.fit(train_pool,
							eval_set= test_pool,
							logging_level='Silent')


class MLP(Model):
	def run_gridsearch(self,cv,cv_score:str)-> None:
		"""
		Performs a gridsearch over the tuning hyperparameters. Determines the 
		best hyperparameters based on the average validation performance 
		calculated over cross-validation folds.
		:param cv: A cross-validarion generator that determines the 
		cross-validation strategy.
		:param cv_score: Measure to evaluate predictions on the validation set. 
		"""
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

				e_stop = EarlyStopping(monitor = params['monitor'], 
									   min_delta = params['min_delta'], 
									   patience = params['iter_patience'], 
									   mode=params['mode'])
				callbacks = [e_stop]
				optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'])

				model = Sequential()
				model.add(Dense(params['num_neurons'],
								input_dim = len(list(self.X_tr)), 
								kernel_initializer = params['weight_init'], 
								activation = params['hidden_activation'], 
								kernel_regularizer = keras.regularizers.l1(params['l1_ratio'])))
				model.add(Dropout(params['dropout_rate']))
				model.add(Dense(1, 
								kernel_initializer = params['weight_init'], 
								activation = params['out_activation'], 
								kernel_regularizer = keras.regularizers.l1(params['l1_ratio'])))

				model.compile(loss = params['loss_func'], 
							  optimizer = optimizer)

				history = model.fit(X_train, 
									y_train, 
									callbacks= callbacks, 
									validation_data = (X_val, y_val), 
									epochs = params['epochs'], 
									batch_size = params['batch_size'], 
									verbose = 0)

				validation_AUC = calc_perf_score(data = X_val, 
												 labels = np.array(y_val.astype('float')), 
												 model = model, 
												 model_name= self.name,
												 score_name = cv_score)
				AUC_val.append(validation_AUC)

			AUC_val = np.mean(AUC_val)

			if AUC_val > best_AUC:
				best_AUC = AUC_val
				self.best_tuning_params = tune

			keras.backend.clear_session()

	def train(self, use_gridsearch_results:bool = True)-> None:
		"""
		Trains a Multilayer Perceptron (MLP) model. 
		:param use_gridsearch_results: Determines whether to use 
		hyperparameters selected through gridsearch
		"""
		params = self.fixed_params.copy()
		if use_gridsearch_results == True:
			self.tuning_params = self.best_tuning_params
			
		params.update(self.tuning_params)

		e_stop = EarlyStopping(monitor = params['monitor'], 
							   min_delta = params['min_delta'], 
							   patience = params['iter_patience'], 
							   mode=params['mode'])
		callbacks = [e_stop]
		optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'])

		model = Sequential()
		model.add(Dense(params['num_neurons'],
						input_dim = len(list(self.X_tr)), 
						kernel_initializer = params['weight_init'], 
						activation = params['hidden_activation'], 
						kernel_regularizer = keras.regularizers.l2(params['l1_ratio'])))
		model.add(Dropout(params['dropout_rate']))
		model.add(Dense(1, 
						kernel_initializer = params['weight_init'], 
						activation = params['out_activation'], 
						kernel_regularizer = keras.regularizers.l2(params['l1_ratio'])))
		model.compile(loss = params['loss_func'], optimizer = optimizer)

		history = model.fit(self.X_tr, 
							self.y_tr, 
							callbacks= callbacks, 
							validation_data = (self.X_te, self.y_te), 
							epochs = params['epochs'], 
							batch_size = params['batch_size'], 
							verbose = 0)

		self.best_model = model