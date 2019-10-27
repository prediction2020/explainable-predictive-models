import numpy as np
import pickle
import yaml
import os
import sys
import time
import keras
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold

from utils.models import *
from utils.dataset import *

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
cfg = yaml.load(open('config.yml', 'r'))

# Assign variables
dataset_name = cfg['dataset']['name']
dataset_path = cfg['dataset']['path']
splits_path = cfg['dataset']['splits path']
number_of_splits = cfg['number of training-test splits']
models_to_train = cfg['models to use']
subsampling_types = cfg['subsampling to use']
test_size = cfg['test size']
fixed_params = cfg['fixed hyperparameters']
tuning_params = cfg['tuning parameters']
cv_fold_count = cfg['cross-validation folds']
cv_score = cfg['cross-validation score']
models_folder = ['models folder path']
params_folder = ['parameters folder path']


########################################################################################
###### GET TRAINING AND TEST DATA ######################################################
########################################################################################

data = ClinicalDataset(name = dataset_name, path = dataset_path)
data.preprocess()
# For the initial assignment of training-test sets you have to specify the test set size, 
# the required number of splits and the path to save the created splits.
#
data.assign_train_test_sets(splits_path, test_size, number_of_splits)

# If the splits have already been created and saved you can just load them as a class
# instance variable using the assign_train_test_sets function with only the path to the 
# splits file
# data.assign_train_test_sets(splits_path)

print('Number of patients in dataset: '+ str(len(data.X)))

#print('Number of patients in training set: '+ str(len(data.X)))
#print('Number of patients in test set: '+ str(len(data.X)))
########################################################################################
###### PERFORM GRID SEARCH AND SAVE BEST PERFORMING MODELS #############################
########################################################################################

# Assign the strategy to choose folds during cross-validation
my_cv = StratifiedKFold(cv_fold_count)

start = time.time()

# Iterate over subsampling types
for subs in subsampling_types:

	# Subsample the training data given the subsampling type. Subsampling has to be 
	# done before iterating over models in order to train all models on the same 
	# patients. This provides better model comparison.
	data.subsample_training_sets(number_of_splits,subs)

	# Iterate over model classes
	for mdl in models_to_train:

		file_suffix = '.h5' if mdl == 'MLP' else '.pkl'

		# Assign model-specific fixed and tuning hyperparameters
		fixed_params = fixed_params[mdl]
		tune_params = tuning_params[mdl]

		# Iterate over splits
		for i in range(number_of_splits):
			
			# Assign main path to save models
			path_to_models_folder = f'{models_folder}/best_{mdl}_model_{subs}_subsampling'

			# Assign path to model file trained on the current split
			path_to_model = f'{path_to_models_folder}/best_{mdl}_model_{subs}_subsampling_split_{i+1}{file_suffix}'

			# Check if the model file path already exists. If, load the model.			
			if os.path.isfile(path_to_model):
				if mdl =='MLP':
					best_model = load_model(path_to_model)
				else:
				  	best_model = pickle.load(open(path_to_model,'rb'))

				print(f'Loaded pre-esxisting {mdl} model with {subs} subsampling trained on split {i+1}.')


			# If the model file path doesn't exist run grid search to find the best
			# model and save it under model file path.
			else:
				print(f'Running grid search using {mdl} on split {i+1} with {subs} subsampling.')

				model = eval(mdl)(name = mdl, 
										dataset = data.splits[i] ,
										fixed_params = fixed_params, 
										tuning_params = tune_params)

				# Don't perform gridsearch if model is GLM
				if mdl == 'GLM':		
					model.train()
				else:
					model.run_gridsearch(cv = my_cv, cv_score = cv_score)
					model.train(use_gridsearch_results = True)

				best_model = model.best_model

				# Create main model folder if it doesn't exist
				if not os.path.exists(path_to_models_folder):
					os.makedirs(path_to_models_folder)

				# Save best model of the current split, model class and subsampling type.
				if mdl =='MLP':
					best_model.save(path_to_model)
				else:
					pickle.dump(best_model, open(path_to_model,'wb'))

				# Print progress 
				if i>0 and i%10 == 0:
					print(str(i*2)+'%s of the process is done.'%('%'))

			# Save best model hyperparameters
			# Assign main path to save hyperparameters
			path_to_params_folder = f'{params_folder}/best_{mdl}_parameters_{subs}_subsampling'

			# Assign path to hyperparameters file
			path_to_params = f'{path_to_params_folder}/best_{mdl}_parameters_{subs}_subsampling_split_{i+1}.json'

			# Create main hyperparameters folder if it doesn't exist. GLM model class is 
			# excluded since there are no hyperparameters tuned for this class.
			if not os.path.exists(path_to_params_folder) and mdl != 'GLM':
				os.makedirs(path_to_params_folder)

			# Save best hyperparameters of the current split, model class and subsampling
			# type.
			if mdl =='MLP':
				with open(path_to_params, "w") as json_file:
					json_file.write(best_model.to_json())
			elif mdl =='Lasso' or mdl == 'Elasticnet' or mdl == 'Catboost':
				json.dump(best_model.get_params(), open(path_to_params, 'w'))

			
end = time.time()

# Print overall computation time
print(f'Overall computation time was {(end-start)/60:.2f} minutes.')
