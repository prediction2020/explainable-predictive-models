import pandas as pd
import numpy as np
import os
import yaml
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing


def mulcolfunc(filename):

	data = pd.read_pickle(filename)
	# Get rid of label if exists
	if 'OU_MRS_3M' in list(data) : data = data.drop('OU_MRS_3M',axis=1)

	parameter_names = list(data)

	# Configuring data types
	num_data = list(data.columns[data.dtypes != 'category' ])

	cols_to_transform = list(data.columns[data.dtypes == 'category' ])
	for i, col in enumerate(cols_to_transform):
		data[col] = data[col].astype('float64')

	if 'AT_ST' in parameter_names:
		data['AT_ST'] = data['AT_ST'] / np.timedelta64(1, 'h')

	# Centering the data
	scaler = preprocessing.StandardScaler()
	data[num_data] = scaler.fit_transform(np.array(data[num_data]))

	# Eliminating vif values bigger than 10
	datam= data

	col_check = 0
	while col_check == 0 :
		vif = np.array([variance_inflation_factor(datam.values, i) for i in range(datam.shape[1])])
		if np.max(vif)>10:
			datam = datam.drop(datam.columns[np.argmax(vif)], axis=1)#np.delete(data.values,np.argwhere(vif>2),axis=1)
		else :
			col_check =1


	parameter_names_m = list(datam)
	vif_table = pd.DataFrame(vif, index=parameter_names_m, columns=['Variance Inflation Factor(VIF)'])
	return vif_table






#### READ CONFIGURATION FILE ##########
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join(str(i) for i in seq)

yaml.add_constructor('!join',join)
cfg = yaml.load(open('config.yml', 'r'))

#### ASSIGN CONFIGURATION VARIABLES #####
dataset_name = cfg['dataset']['name']
dataset_path = cfg['dataset']['path']
to_save_path = 'data_analysis/multicollinearity_analysis/'



if not os.path.exists(to_save_path):
    os.makedirs(to_save_path)

vif = mulcolfunc(dataset_path)
vif.to_csv(to_save_path+dataset_name+'_multicollinearity.csv',float_format= '%2.2f')


