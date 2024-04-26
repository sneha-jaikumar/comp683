import pandas as pd
import os
from os import listdir
import anndata
import scanpy as sc
import numpy as np
from itertools import repeat
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
from numpy import random
from sklearn.metrics import classification_report
from sklearn.linear_model import ElasticNet


def pseudobulk_feats(data = None, meta_label = 'meta_label', end_range = None):
    """
    Computes functional features (median expression of metaclusters within a sample)
    Parameters
    ----------
    data: pandas.DataFrame
    meta_label: str
        string referencing column containing metacluster label ID
    k: int
        number of highly_variable_genes of the input data
    Returns
    ----------
    pseudobulk_feats: pandas.DataFrame
        dataframe containing functional features for each sample
    """
    d = data.groupby(['sample_id', meta_label])
    median_data = {}
    for name, group in d:
        median_expression = group.iloc[:, :end_range].median()
        #print(median_expression)
        median_data[name] = median_expression
    pseudobulk_feats = pd.DataFrame.from_dict(median_data, orient='index').unstack().fillna(0)
    pseudobulk_feats.columns = ['{}_exp_{}'.format(i, j) for i, j in pseudobulk_feats.columns]
    return pseudobulk_feats

# 35450 × 3000
adata_merge = sc.read_h5ad('/Users/snehajaikumar/CompScience/comp683/merged_adata.h5ad')
sc.pp.highly_variable_genes(adata_merge, flavor = 'seurat', n_top_genes = 2000, subset = True)
data_mod = pd.DataFrame(adata_merge.X, columns = adata_merge.var_names, index = adata_merge.obs_names)
data_mod['sample_id'] = adata_merge.obs['shortname'].copy()
data_mod['cluster_label'] = adata_merge.obs['leiden'].copy()
data_mod['age'] = adata_merge.obs['age'].copy()

#pseudobulk preprocessing
result_1 = pseudobulk_feats(data = data_mod, meta_label = 'cluster_label', end_range = 2000)
column_labels = list(result_1.columns)
unique_genes_ordered = list(OrderedDict.fromkeys(element.split("_")[0] for element in column_labels))
result = result_1.astype(float)
result_temp = result.apply(lambda x: np.log(x+1), axis=1)
reshaped_data = result_temp.squeeze()
reshaped_data = reshaped_data.values.reshape(48, -1, 21)
averages = np.mean(reshaped_data, axis=2)
averages_df = pd.DataFrame(averages, index=result.index, columns=range(0, 2000))
averages_df.columns = unique_genes_ordered
filtered_pseudo_temp = averages_df#[common_elements]
#Standard scaler
scaler = StandardScaler()
scaler.fit(filtered_pseudo_temp)
scaled_data = scaler.transform(filtered_pseudo_temp)
filtered_pseudo = pd.DataFrame(scaled_data, columns = filtered_pseudo_temp.columns)
#Add age column back to df now
filtered_pseudo["Age"] = result_1.index.str.split('_').str[1]
#Drop age column
pseudo_bulk_1 =  filtered_pseudo.drop(["Age"], axis=1)
#Add age column back again
pseudo_bulk_1.insert(0,"Age",filtered_pseudo.Age)

#View Hammond pseudobulk
print(pseudo_bulk_1)

#_________________________________________________
# 17295 × 677
adata_merge = sc.read_h5ad('/Users/snehajaikumar/CompScience/comp683/adata_merge_fetuses_leiden.h5ad')
data_mod = pd.DataFrame(adata_merge.X, columns = adata_merge.var_names, index = adata_merge.obs_names)
data_mod['sample_id'] = adata_merge.obs['shortname'].copy()
data_mod['cluster_label'] = adata_merge.obs['leiden'].copy()
data_mod['age'] = adata_merge.obs['age'].copy()

result_1 = pseudobulk_feats(data = data_mod, meta_label = 'cluster_label', end_range = 677)
column_labels = list(result_1.columns)
unique_genes_ordered = list(OrderedDict.fromkeys(element.split("_")[0] for element in column_labels))
result = result_1.astype(float)
result_temp = result.apply(lambda x: np.log(x+1), axis=1)
# 239 x 10800, 16 clusters
#Take the average of genes across all 16 clusters
reshaped_data = result_temp.squeeze()
reshaped_data = reshaped_data.values.reshape(239, -1, 16)
averages = np.mean(reshaped_data, axis=2)
averages_df = pd.DataFrame(averages, index=result.index, columns=range(0, 675))
averages_df.columns = unique_genes_ordered
filtered_pseudo_temp = averages_df#[common_elements]
# Standard scaler
scaler = StandardScaler()
scaler.fit(filtered_pseudo_temp)
scaled_data = scaler.transform(filtered_pseudo_temp)
filtered_pseudo = pd.DataFrame(scaled_data, columns = filtered_pseudo_temp.columns)
# Add age column back to df now
filtered_pseudo["Age"] = result_1.index.str.split('_').str[1]
# Drop age column
pseudo_bulk_1 =  filtered_pseudo.drop(["Age"], axis=1)
# Add age column back again
pseudo_bulk_1.insert(0,"Age",filtered_pseudo.Age)

#View Fetus Pseudobulk
print(pseudo_bulk_1)
