import numpy as np
import pickle
import yaml
import os
import sys
import keras
import shap
import pandas as pd
from keras.models import load_model 
from vis.utils import utils
import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.keras.graph as inkgraph

from clinical_predictive_models import *
from helper import *
from subsampling import *
from BP_keras_utils import *

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
cfg = yaml.load(open('config.yml', 'r'), Loader= yaml.Loader)

# Assign variables
dataset_name = cfg['dataset']['name']
dataset_path = cfg['dataset']['path']
splits_path = cfg['dataset']['splits path']
number_of_splits = cfg['number of splits']
models_to_train = cfg['models to use']
subsampling_types = cfg['subsampling to use']
performance_scores = cfg['final performance measures']
test_size = cfg['test size']
save_models = cfg['save options']['models path']
save_scores = cfg['save options']['scores path']
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
# splits = data.assign_train_test_sets(splits_path, test_size, number_of_splits)

# If the splits have already been created and saved you can just load them into the 
# splits variable using the assign_train_test_sets function with only the path to the 
# splits file
splits = data.assign_train_test_sets(splits_path)

print('number of patients: '+ str(len(data.X)))


#########################################################################################
###### EVALUATE MODELS  AND SAVE PERFORMANCE SCORES #####################################
#########################################################################################

# Check if the scores folder path to save already exists. If not, create folder.
if not os.path.exists(save_scores):
	os.makedirs(save_scores)

# Iterate over subsampling types
for subs in subsampling_types:
	sets_to_use = data.subsample_training_sets(number_of_splits,subs)

	# Iterate over models
	for mdl in models_to_train:
		file_suffix = '.h5' if mdl == 'MLP' else '.pkl'

		# Iterate over performance scores
		for scr in performance_scores:
			path_to_scores =  save_scores+'/'+mdl+'_'+scr+'_scores_'+subs+'_subsampling.csv'

			# Check if the scores file path already exists. If it exists, return to previous loop
			if not os.path.isfile(path_to_scores):
				tmp_scores = np.zeros((number_of_splits,2))

				# Iterate over splits
				for i in range(number_of_splits):
					path_to_model = save_models+'/best_'+mdl+'_models_'+subs+'_subsampling/best_'+mdl+'_model_'+subs+'_subsampling_run_'+str(i+1)+file_suffix
					model_obj = model(mdl, sets_to_use[i], default_params = {'out_activation':'sigmoid'})
					if mdl == 'MLP':
						model_obj.best_model = load_model(path_to_model)
					else : 
						model_obj.best_model = pickle.load(open(path_to_model,'rb'))
					score_tr = model_obj.get_performance_score(model_obj.X_tr, model_obj.y_tr,score_name=scr)
					score_te = model_obj.get_performance_score(model_obj.X_te, model_obj.y_te,score_name=scr)
					#print(score_tr,score_te)
					tmp_scores[i,0] = score_tr
					tmp_scores[i,1] = score_te

					keras.backend.clear_session()

				df_scores = pd.DataFrame(tmp_scores, columns=['training','test'])
				df_scores.to_csv(path_to_scores)
				#pickle.dump(tmp_scores, open(path_to_scores,'wb'))

				print('Saved '+scr+' scores for '+mdl+' model with '+subs+' subsampling.')

#############################################################################################
###### CALCULATE WEIGHTS, SHAP AND BACKPROPOGATION VALUES ###################################
#############################################################################################

# Check if the weights folder path to save already exists. If not, create folder.
if not os.path.exists(save_weights):
	os.makedirs(save_weights)	

# Iterate over subsampling types
for subs in subsampling_types:
	sets_to_use = data.subsample_training_sets(number_of_splits,subs)

	# Iterate over models
	for mdl in models_to_train:

		# Get model weights for linear classifiers directly:
		if mdl in ['GLM', 'Lasso', 'Elasticnet']:
			path_to_weights = save_weights+'/best_'+mdl+'_weights_'+subs+'_subsampling.csv'
			if not os.path.isfile(path_to_weights):
				tmp_weights = np.zeros((number_of_splits,len(data.cols)-1))

				for i in range(number_of_splits):
					path_to_model = save_models+'/best_'+mdl+'_models_'+subs+'_subsampling/best_'+mdl+'_model_'+subs+'_subsampling_run_'+str(i+1)+'.pkl'
					model_obj = model(mdl, sets_to_use[i],default_params = {'out_activation':'sigmoid'})
					model_obj.best_model = pickle.load(open(path_to_model,'rb'))

					weights = ["{:.5f}".format(weight) for weight in model_obj.best_model.coef_[0]]
					tmp_weights[i,:] = weights

				df_weights = pd.DataFrame(tmp_weights, columns= list(model_obj.X_tr))
				df_weights.to_csv(path_to_weights)

				print('Saved weights of '+mdl+' model with '+subs+' subsampling.')

		# Calculate shap values for the non-linear Catboost classifier
		elif mdl == 'Catboost':
			path_to_shaps = save_weights+'/best_'+mdl+'_shap_values_'+subs+'_subsampling.csv'
			if not os.path.isfile(path_to_shaps):
				tmp_shaps = np.zeros((number_of_splits,len(data.cols)-1))

				for i in range(number_of_splits):
					path_to_model = save_models+'/best_'+mdl+'_models_'+subs+'_subsampling/best_'+mdl+'_model_'+subs+'_subsampling_run_'+str(i+1)+'.pkl'
					model_obj = model(mdl, sets_to_use[i],default_params = {'out_activation':'sigmoid'})
					model_obj.best_model = pickle.load(open(path_to_model,'rb'))

					cats = model_obj.X_tr.columns[model_obj.X_tr.dtypes=='category']
					explainer = shap.TreeExplainer(model_obj.best_model)
					shap_values = explainer.shap_values(cat.Pool(model_obj.X_tr,
																 model_obj.y_tr, 
																 cat_features = [list(model_obj.X_tr).index(cats[i]) for i in range(len(cats))]))
					shap_values_mean_over_samples = ["{:.5f}".format(shap) for shap in np.mean(shap_values,axis=0)]
					tmp_shaps[i,:] = shap_values_mean_over_samples
					
				df_shaps = pd.DataFrame(tmp_shaps, columns= list(model_obj.X_tr))
				df_shaps.to_csv(path_to_shaps)

				print('Saved shap values of '+mdl+' model with '+subs+' subsampling.')
		
		# Calculate back-propagation values for the non-linear MLP classifier			
		elif mdl == 'MLP':
			#path_to_bp = save_weights+'/best_'+mdl+'_bp_values_'+subs+'_subsampling.csv'
			path_to_dt_score_based = save_weights+'/best_'+mdl+'_score_based_averaged_dt_values_'+subs+'_subsampling.csv'
			path_to_dt_simple = save_weights+'/best_'+mdl+'_simple_averaged_dt_values_'+subs+'_subsampling.csv'
			tmp_dt_score_based = np.zeros((number_of_splits,len(data.cols)-1))
			tmp_dt_simple = np.zeros((number_of_splits,len(data.cols)-1))
			#tmp_bp = np.zeros((number_of_splits,len(data.cols)-1))

			for i in range(number_of_splits):
				# Load model
				path_to_model = save_models+'/best_'+mdl+'_models_'+subs+'_subsampling/best_'+mdl+'_model_'+subs+'_subsampling_run_'+str(i+1)+'.h5'
				model_obj = model(mdl, sets_to_use[i],default_params = {'out_activation':'sigmoid'})
				model_obj.best_model = load_model(path_to_model)

				# Predict training and test probabilities
				test_probs = model_obj.predict_probability(model_obj.X_te)
				train_probs = model_obj.predict_probability(model_obj.X_tr)

				# Set last layer activation to linear. If this swapping is not done, the 
				# results might be suboptimal
				stripped_model = last_to_linear(model_obj.best_model)

				# Calculate class weights
				train_input_weights = train_probs
				train_input_weights[np.where(model_obj.y_tr == 0)] = 1-train_input_weights[np.where(model_obj.y_tr == 0)]

				class_idx = 0 # if the activation of last layer was sigmoid
				last_layer_idx = utils.find_layer_idx(model_obj.best_model, 'dense_2')

				#if not os.path.isfile(path_to_bp):
			#		
					# Calculate global gradients of all patients (backprop)					
			#		wavg_train_grads = wavg_saliency_grads(stripped_model,class_idx=class_idx,inputs=model_obj.X_tr,
	        #                           input_weights=train_input_weights,last_layer_idx=last_layer_idx,
	        #                           backprop_modifier='guided')
			#		tmp_bp[i,:] = wavg_train_grads

				

				if not os.path.isfile(path_to_dt_score_based):
					
					# Calculate global gradients of all patients (deep taylor)
					seed_input = model_obj.X_tr.values
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

					tmp_dt_score_based[i,:] = score_avg_analysis
					tmp_dt_simple[i,:] = avg_analysis

				keras.backend.clear_session()

			#df_bp = pd.DataFrame(tmp_bp, columns= list(model_obj.X_tr))
			#df_bp.to_csv(path_to_bp)

			#print('Saved backpropagation values of '+mdl+' model with '+subs+' subsampling.')

			df_dt_score_based = pd.DataFrame(tmp_dt_score_based, columns= list(model_obj.X_tr))
			df_dt_score_based.to_csv(path_to_dt_score_based)

			df_dt_simple = pd.DataFrame(tmp_dt_simple, columns= list(model_obj.X_tr))
			df_dt_simple.to_csv(path_to_dt_simple)

			print('Saved deep taylor values of '+mdl+' model with '+subs+' subsampling.')
					
                                       



			








def models_comparison(score,model_names,sub_type,score_name,path):
    score_m = np.asarray([score[m][:,1] for m in model_names])
    significance =[]
    labels= []
    #fig = plt.figure(figsize=(10,6))
    #plt.boxplot(score_m.T)
    #plt.xticks([1,2,3,4],model_names)
    #plt.ylabel('Average test score')
    #plt.title('Model Comparison on '+score_name+' score distributions ('+sub_type.replace('none','no')+' subsampling)')
    for mdl1,mdl2 in itertools.combinations(model_names,2):
        _, p_val = wilcoxon(score[mdl1][:,1],score[mdl2][:,1])
        significance.append(float(format(p_val, '.4f')))
        labels.append(mdl1+'_'+ mdl2)

    # BEAUTIFY! #
    #for i in range(len(model_names)):
    #    for j in range(len(model_names)):
    #        if i<j:
    #            y_change = 0.04 if j==i+1 else 0.17 if j==i+3 else 0.09 if (j==i+2)&(i==0) else 0.15
    #            y, h, col = np.concatenate((score[mdl1][:,1],score[mdl2][:,1])).max() + y_change, 0.01, 'k'
    #            if p_val<0.05 and p_val>0.01:
    #                plt.plot([i+1.05, i+1.05, j+0.95, j+0.95], [y, y+h, y+h, y], lw=1.5, c=col)
    #                plt.text(((i+1)+(j+1))*.5, y+h, '*', ha='center', va='bottom', color=col)
    #            elif p_val<0.01 and p_val>0.001:
    #                plt.plot([i+1.05, i+1.05, j+0.95, j+0.95], [y, y+h, y+h, y], lw=1.5, c=col)
    #                plt.text(((i+1)+(j+1))*.5, y+h, '**', ha='center', va='bottom', color=col)
    #            elif p_val<0.001:
    #                plt.plot([i+1.05, i+1.05, j+0.95, j+0.95], [y, y+h, y+h, y], lw=1.5, c=col)
    #                plt.text(((i+1)+(j+1))*.5, y+h, '***', ha='center', va='bottom', color=col)

    p_df = pd.DataFrame(significance ,columns=['test_AUC_p_value'],index= labels)
    p_df.to_csv('/models_comparison_on_test_'+score_name+'_scores_'+sub_type+'_subsampling.csv')
    #fig.savefig(path+'/models_comparison_on_test_'+score_name+'_scores_'+sub_type+'_subsampling.png',dpi=199)
    #plt.close(fig)