import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import os
import yaml
import seaborn as sns
from utils.dataset import ClinicalDataset

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

# Assign variables to use
dataset_name = cfg['dataset']['name']
dataset_path = cfg['dataset']['path']
features_dict = cfg['features']
folder_to_save = 'data_analysis/data_statistics/'


########################################################################################
###### PLOT CLINICAL FEATURES DISTRIBUTIONS ############################################
########################################################################################

if not os.path.exists(folder_to_save):
    os.makedirs(folder_to_save)

# Load dataset
data = ClinicalDataset(name = dataset_name, path = dataset_path)
df = data.df

# Print categorical variable ratios:
for cat in data.cat_data:
    print(df[cat].value_counts())

# Print median and iqr of numerical variables:
for num in data.num_data:
    print(f'{num}: median = {np.median(df[num])}, iqr = {iqr(df[num])}')


df_new=df.copy()
for i, col in enumerate(data.cat_data):
    df_new[col] = df[col].astype('int64')
df_m= df_new.as_matrix(columns=data.cat_data)
cat_dist = np.mean(df_m, axis=0)
fig1 = plt.figure(figsize=(15,6))
plt.plot(cat_dist,'o')
plt.ylabel('Mean',size=14)
plt.axhline(0.5,ls='--', color='gray')
plt.xticks(np.arange(len(data.cat_data)),data.cat_data,size=14)
plt.title("Distribution of Categorical Parameters",size=18)
fig1.savefig(folder_to_save+'/'+dataset_name+'_dist_of_categorical_data.png')


fig2, axs = plt.subplots(1,len(data.num_data),figsize=(12,8))
for i, num_var in enumerate(data.num_data):
    axs[i].boxplot(df[num_var])
    axs[i].set_title('Distribution of %s'%(features_dict[num_var]))
    axs[i].set_xticklabels([features_dict[num_var]])

fig2.savefig(folder_to_save+'/'+dataset_name+'_dist_of_numerical_data.png')


df_cat = df.drop(list(data.num_data),axis=1)
for i, col in enumerate(data.cat_data):
    df_cat[col] = df_cat[col].astype('int64')

fig3 = plt.figure(figsize=(15,6))
sns.countplot(x='variable',hue='value',data=pd.melt(df_cat))
plt.title('Histogram of categorical variables')
fig3.savefig(folder_to_save+'/'+dataset_name+'_hist_of_categorical_data.png')


fig4, axs = plt.subplots(1,len(data.num_data),figsize=(15,6))
for i, num_var in enumerate(data.num_data):
    sns.distplot(df[num_var], ax=axs[i])
    axs[i].set_title('Histogram of %s'%(features_dict[num_var]))
fig4.savefig(folder_to_save+'/'+dataset_name+'_hist_of_numerical_data.png')
