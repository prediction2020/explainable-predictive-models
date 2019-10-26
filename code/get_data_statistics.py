import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import os
import yaml

import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

import seaborn as sns
from scipy.stats import iqr

#### READ CONFIGURATION FILE ##########
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join(str(i) for i in seq)

yaml.add_constructor('!join',join)
cfg = yaml.load(open('config.yml', 'r'))

#### ASSIGN CONFIGURATION VARIABLES #####
dataset_name = cfg['dataset']['name']
dataset_path = cfg['dataset']['path']
folder_to_save = 'data_analysis/data_statistics/'

if not os.path.exists(folder_to_save):
    os.makedirs(folder_to_save)

features_dict = {'AD_NIH':'NIHSS','AS_TS_MCA':'localisation','AT_LY':'thrombolysis treatment','DG_SEX':'gender',
         'RF_AH':'arterial hypertension','RF_DM':'diabetes','RF_HC':'hypercholesterol','RF_SM':'smoker',
         'DG_AG':'age','RF_SM_F':'former smoker','CH':'cardiac history', 'AT_ST':'onset-to-treatment time' ,'DWI': 'acute DWI lesion volume'}

# For all patients
df = pd.read_pickle(dataset_path)


if 'AT_ST' in list(df):
    df['AT_ST'] = df['AT_ST'] / np.timedelta64(1, 'h')
    while df['AT_ST'].max()>100:
        df = df.drop(df['AT_ST'].idxmax())
cat = df.columns[df.dtypes=='category']
nums = df.columns[df.dtypes!='category']

# Print categorical variable ratios:
for col in cat:
    print(df[col].value_counts())

# Print median and iqr of numerical variables:
for col in nums:
    print(f'{col}: median = {np.median(df[col])}, iqr = {iqr(df[col])}')

print(a)

df_new=df.copy()
for i, col in enumerate(cat):
    df_new[col] = df[col].astype('int64')
df_m= df_new.as_matrix(columns=cat)
cat_dist = np.mean(df_m, axis=0)
fig1 = plt.figure(figsize=(15,6))
plt.plot(cat_dist,'o')
plt.ylabel('Mean',size=14)
plt.axhline(0.5,ls='--', color='gray')
plt.xticks(np.arange(len(cat)),cat,size=14)
plt.title("Distribution of Categorical Parameters",size=18)
fig1.savefig(folder_to_save+'/'+dataset_name+'_dist_of_categorical_data.png')


fig2, axs = plt.subplots(1,len(nums),figsize=(12,8))
for i, num_var in enumerate(nums):
    #df_cat = df.drop(num_var,axis=1)
    axs[i].boxplot(df[num_var])
    axs[i].set_title('Distribution of %s'%(features_dict[num_var]))
    axs[i].set_xticklabels([features_dict[num_var]])

fig2.savefig(folder_to_save+'/'+dataset_name+'_dist_of_numerical_data.png')


df_cat = df.drop(list(nums),axis=1)
for i, col in enumerate(cat):
    df_cat[col] = df_cat[col].astype('int64')

fig3 = plt.figure(figsize=(15,6))
sns.countplot(x='variable',hue='value',data=pd.melt(df_cat))
plt.title('Histogram of categorical variables')
fig3.savefig(folder_to_save+'/'+dataset_name+'_hist_of_categorical_data.png')


fig4, axs = plt.subplots(1,len(nums),figsize=(15,6))
for i, num_var in enumerate(nums):
    sns.distplot(df[num_var], ax=axs[i])
    axs[i].set_title('Histogram of %s'%(features_dict[num_var]))
fig4.savefig(folder_to_save+'/'+dataset_name+'_hist_of_numerical_data.png')
