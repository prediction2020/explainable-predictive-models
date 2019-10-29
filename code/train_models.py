import numpy as np
import pickle
import yaml
import os
import time
import json
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
dataset_name = cfg['dataset name']
dataset_path = cfg['data path']
splits_path = cfg['splits path']
number_of_splits = cfg['number of splits']
models_to_train = cfg['models to use']
subsampling_types = cfg['subsampling to use']
test_size = cfg['test size']
fixed_parameters = cfg['fixed hyperparameters']
use_gridsearch = cfg['use gridsearch']
tuning_parameters = cfg['tuning hyperparameters']
cv_fold_count = cfg['cross-validation folds']
cv_score = cfg['cross-validation score']
models_folder = cfg['models folder path']
params_folder = cfg['parameters folder path']

########################################################################################
###### GET TRAINING AND TEST DATA ######################################################
########################################################################################

data = ClinicalDataset(name = dataset_name, path = dataset_path)
data.preprocess()
# For the initial assignment of training-test sets you have to specify the test set size, 
# the required number of splits and the path to save the created splits.
#
data.assign_train_test_splits(path = splits_path, test_size = test_size, splits = number_of_splits)

# If the splits have already been created and saved you can just load them as a class
# instance variable using the assign_train_test_sets function with only the path to the 
# splits file
# Data.assign_train_test_splits(splits_path)

print('Number of patients in dataset: '+ str(len(data.X)))


########################################################################################
###### PERFORM GRID SEARCH AND SAVE BEST PERFORMING MODELS #############################
########################################################################################

# Create main models folder if it doesn't exist
if not os.path.exists(models_folder):
	os.makedirs(models_folder)

# Create main hyperparameters folder if it doesn't exist. GLM model class is 
# excluded since there are no hyperparameters tuned for this class.
if not os.path.exists(params_folder):
	os.makedirs(params_folder)

# Assign the strategy to choose folds during cross-validation
my_cv = StratifiedKFold(cv_fold_count, random_state=21)

start = time.time()

# Iterate over subsampling types
for subs in subsampling_types:

	# Subsample the training data given the subsampling type. In order to provide 
	# comparable results, the seed was fixed for random sampling.
	data.subsample_training_set(number_of_splits = number_of_splits,subsampling_type = subs)

	# Iterate over model classes
	for mdl in models_to_train:

		file_suffix = '.h5' if mdl == 'MLP' else '.pkl'

		# Assign model-specific fixed and tuning hyperparameters
		fixed_params = fixed_parameters[mdl]
		tune_params = tuning_parameters[mdl]

		# Iterate over splits
		for i in range(number_of_splits):
			
			# Assign path to model file to be trained on the current split
			path_to_model = f'{models_folder}/{mdl}_model_{subs}_subsampling_split_{i+1}{file_suffix}'

			# Create model instance for the current split 
			model = eval(mdl)(name = mdl, 
							  dataset = data.splits[i] ,
							  fixed_params = fixed_params, 
							  tuning_params = tune_params)

			# Check if the model file path already exists.			
			if os.path.isfile(path_to_model):
				print(f'{mdl} model with {subs} subsampling trained on split {i+1} already exists.')


			# If the model file path doesn't exist:
			else:
				# 1. If'use gridsearch' option is set to True; run grid search to 
				# find the best hyperparameters, train model on those hyperparameters
				# (except for GLM, which doesn't have any tunable hyperparameters).
				if use_gridsearch == True and mdl != 'GLM':			
					print(f'Running grid search using {mdl} on split {i+1} with {subs} subsampling.')
					model.run_gridsearch(cv = my_cv, cv_score = cv_score)
					print(f'Training {mdl} on split {i+1} with {subs} subsampling.')
					model.train(use_gridsearch_results = True)

				# 2. If 'use gridsearch' option is set to False; train model on the
				# defined fixed hyperparameters.
				else:
					print(f'Training {mdl} on split {i+1} with {subs} subsampling.')
					model.train(use_gridsearch_results = False)

				# Save best model on the current split.
				model.save_model(path_to_model)
				

				# Save best tuning hyperparameters (except GLM)
				if mdl != 'GLM':
					# Assign path to hyperparameters file
					path_to_params = f'{params_folder}/best_{mdl}_parameters_{subs}_subsampling_split_{i+1}.json'

					# Save best hyperparameters on the current split
					json.dump(model.best_tuning_params, open(path_to_params, 'w'))
		
end = time.time()

# Print overall computation time
print(f'Overall computation time was {(end-start)/60:.2f} minutes.')
