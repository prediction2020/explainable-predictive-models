import numpy as np
import pickle
import yaml
import os
import sys
import time
import keras
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold

from clinical_predictive_models import *
from subsampling import *

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
number_of_splits = cfg['number of splits']
models_to_train = cfg['models to use']
subsampling_types = cfg['subsampling to use']
test_size = cfg['test size']
default_params = cfg['hyperparameters']
tuning_params = cfg['tuning parameters']
cv_fold_count = cfg['cv folds']
cv_score = cfg['validation score']
save_models = cfg['save options']['models path']
save_params = cfg['save options']['params path']
save_weights = cfg['save options']['weights path']


########################################################################################
###### GET TRAINING AND TEST DATA ######################################################
########################################################################################

data = clinical_dataset(dataset_name)
data.load_data(dataset_path)
data.preprocess()
# For the initial assignment of training-test sets you have to specify the test set size, 
# the required number of splits and the path to save the created splits.
#
splits = data.assign_train_test_sets(splits_path, test_size, number_of_splits)

# If the splits have already been created and saved you can just load them into the 
# splits variable using the assign_train_test_sets function with only the path to the 
# splits file
# splits = data.assign_train_test_sets(splits_path)

print('number of patients: '+ str(len(data.X)))


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
	splits_to_use = data.subsample_training_sets(number_of_splits,subs)

	# Iterate over model classes
	for mdl in models_to_train:

		file_suffix = '.h5' if mdl == 'MLP' else '.pkl'

		# Assign fixed and tuning hyperparameters
		def_params = default_params[mdl]
		tune_params = tuning_params[mdl]

		# Iterate over splits
		for i in range(number_of_splits):
			
			# Assign main path to save models
			path_to_folder = save_models + '/best_' + mdl + '_models_' + subs + '_subsampling'

			# Assign path to model file
			path_to_model = path_to_folder + '/best_'+ mdl + '_model_' + subs + '_subsampling_run_' + str(i+1) + file_suffix

			# Check if the model file path already exists. If, load the model.			
			if os.path.isfile(path_to_model):
				if mdl =='MLP':
					best_model = load_model(path_to_model)
				else:
				  	best_model = pickle.load(open(path_to_model,'rb'))


			# If the model file path doesn't exist run grid search to find the best
			# model and save it under model file path.
			else:
				print('Running grid search on %s model with %s subsampling for split %i.'
						%(mdl,subs,i+1))
				grid_search = eval(mdl + '_cv')(mdl, splits_to_use[i] ,def_params,
								   tune_params)
				grid_search.run(my_cv, cv_score)
				best_model = grid_search.best_model

				# Create main model folder if it doesn't exist
				if not os.path.exists(path_to_folder):
					os.makedirs(path_to_folder)

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
			path_to_params_folder = save_params + '/best_' + mdl + '_params_' + subs + '_subsampling'

			# Assign path to hyperparameters file
			path_to_params = path_to_params_folder + '/best_' + mdl + '_params_' + subs + '_subsampling_run_' + str(i+1) + '.json'

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
