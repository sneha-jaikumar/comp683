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
import re
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression



np.random.seed(42)

def accuracy(y_true, y_pred):
    
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score
    
    """
    
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == yp:
            
            correct_predictions += 1
    
    #returns accuracy
    return correct_predictions / len(y_true)

# For fetus pseudobulk
def convert_gest_age_to_days(gest_age_str):
    weeks, days = map(int, gest_age_str.split('+'))
    return weeks * 7 + days

# For fetus bulk
def convert_age_to_days(age_str):
    digits = re.findall(r'(\d+)\s*(?:weeks)?(?![^(]*\))', age_str)
        
    # If there are no digits, return None
    if not digits:
        return None
    
    # Convert the first digit to an integer (weeks)
    weeks = int(digits[0])
    
    # If there are two digits (weeks + days), convert the second digit to an integer (days)
    if len(digits) == 2:
        days = int(digits[1])
    else:
        days = 0
    
    # Calculate the total number of days
    return weeks * 7 + days



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
result_1_m = pseudobulk_feats(data = data_mod, meta_label = 'cluster_label', end_range = 2000)
column_labels = list(result_1_m.columns)
unique_genes_ordered = list(OrderedDict.fromkeys(element.split("_")[0] for element in column_labels))
result = result_1_m.astype(float)
result_temp_m = result.apply(lambda x: np.log(x+1), axis=1)
reshaped_data = result_temp_m.squeeze()
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
filtered_pseudo["Age"] = result_1_m.index.str.split('_').str[1]



#  LOAD IN BULK DATASET
all_df = pd.read_csv('reemst_bulk.csv')
print(all_df['Age'])
#get only the genes shared between bulk and pseudobulk (this removes cond and age)
common_elements_m = list(set(list(all_df.columns)).intersection(unique_genes_ordered))
filtered_bulk = all_df[common_elements_m] 
#preprocessing, log and scale
filtered_bulk = filtered_bulk.astype(float)
log_bulk_temp = filtered_bulk.apply(lambda x: np.log(x+1), axis=1)
scaler = StandardScaler()
scaler.fit(log_bulk_temp)
scaled_bulk = scaler.transform(log_bulk_temp)
final_bulk_reemst = pd.DataFrame(scaled_bulk, columns = log_bulk_temp.columns)
# final_bulk_all.to_csv("final_bulk_all.csv")
expected_list_all = [1 if 'ELS' in condition else 0 for condition in all_df['Condition']]
ctl_indices_all = [index for index, value in enumerate(expected_list_all) if value == 0]
els_indices_all = [index for index, value in enumerate(expected_list_all) if value == 1]
# now go back and add age again
final_bulk_reemst.insert(0,"Age", all_df["Age"])
print(final_bulk_reemst)



# Filter pseudobulk dataset
filtered_pseudobulk_hammond = filtered_pseudo[common_elements_m]
# Now go back and add age again
filtered_pseudobulk_hammond.insert(0,"Age",filtered_pseudo["Age"])
print(filtered_pseudobulk_hammond)

#_________________________________________________
# FETUS DATASET
# 17295 × 677
adata_merge = sc.read_h5ad('/Users/snehajaikumar/CompScience/comp683/adata_merge_fetuses_leiden.h5ad')
data_mod = pd.DataFrame(adata_merge.X, columns = adata_merge.var_names, index = adata_merge.obs_names)
data_mod['sample_id'] = adata_merge.obs['shortname'].copy()
data_mod['cluster_label'] = adata_merge.obs['leiden'].copy()
data_mod['age'] = adata_merge.obs['age'].copy()
result_1_h = pseudobulk_feats(data = data_mod, meta_label = 'cluster_label', end_range = 677)
column_labels = list(result_1_h.columns)
unique_genes_ordered = list(OrderedDict.fromkeys(element.split("_")[0] for element in column_labels))
result = result_1_h.astype(float)
result_temp_h = result.apply(lambda x: np.log(x+1), axis=1)
# 239 x 10800, 16 clusters
#Take the average of genes across all 16 clusters
reshaped_data = result_temp_h.squeeze()
reshaped_data = reshaped_data.values.reshape(239, -1, 16)
averages = np.mean(reshaped_data, axis=2)
averages_df = pd.DataFrame(averages, index=result.index, columns=range(0, 675))
averages_df.columns = unique_genes_ordered
filtered_pseudo_temp = averages_df
# Standard scaler
scaler = StandardScaler()
scaler.fit(filtered_pseudo_temp)
scaled_data = scaler.transform(filtered_pseudo_temp)
filtered_pseudo = pd.DataFrame(scaled_data, columns = filtered_pseudo_temp.columns)
#Add gestational ages
df_metadata = pd.read_excel('fetuses_metadata.xlsx', usecols=['sample_name', 'Gestational week','Gestational age (weeks + days)'])
filtered_pseudo['sample_name'] = result_temp_h.index.str.split('_').str[0]
merged_df = pd.merge(filtered_pseudo, df_metadata, on='sample_name', how='left')
merged_df['Gestational age (days)'] = merged_df['Gestational age (weeks + days)'].apply(convert_gest_age_to_days)



# NOW ADD BULK FETUSES DATASET
df = pd.read_csv("GSE107128_human_fetuses_bulk.csv")
df_transposed = df.T
df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed.drop(df_transposed.index[0])
df_transposed['Age'] = df_transposed.index
df_transposed.reset_index(drop=True, inplace=True)
# Modify Age column to be numbers
df_transposed['Age'] = df_transposed['Age'].apply(convert_age_to_days)
grouped_df = df_transposed.groupby(axis=1, level=0).mean()
common_elements_h = list(set(list(grouped_df.columns)).intersection(unique_genes_ordered)) #This won't include Age
# filter bulk to only include common elements
filtered_bulk_fetus = grouped_df[common_elements_h]
filtered_bulk_fetus = filtered_bulk_fetus.astype(float)
log_bulk_temp_1 = filtered_bulk_fetus.apply(lambda x: np.log(x+1), axis=1)
scaler = StandardScaler()
scaler.fit(log_bulk_temp_1)
scaled_bulk = scaler.transform(log_bulk_temp_1)
final_bulk_fetus = pd.DataFrame(scaled_bulk, columns = log_bulk_temp_1.columns)
# Now go back and add age again
final_bulk_fetus.insert(0,"Age",grouped_df["Age"])
final_bulk_fetus['Age'] = final_bulk_fetus['Age'].astype(int)
print(final_bulk_fetus)



# Filter pseudobulk to only include common elements
filtered_pseudobulk_fetus = merged_df[common_elements_h]
# Now go back and add age again
filtered_pseudobulk_fetus.insert(0,"Age",merged_df["Gestational age (days)"])
print(filtered_pseudobulk_fetus)




# PCA
# X, y = filtered_pseudobulk_hammond.drop(["Age"], axis=1), filtered_pseudobulk_hammond.Age
# pca = PCA(n_components=2)
# Xt = pca.fit_transform(X)
# df_temp = pd.DataFrame(dict(PC1 = Xt[:, 0], PC2 = Xt[:, 1], Age = y))
# colors_dict = {'E14': '#660066',
#                'P4': '#0066cc',
#                'P5': '#ff9900',
#                'P30': '#009933',
#                'P100': '#cc0000',
#                'Old': '#9933ff'
#                }
# sns.scatterplot(x=df_temp['PC1'], y=df_temp['PC2'], hue=df_temp['Age'], palette = colors_dict)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Mice Pseudobulk Data: 2D PCA")
# plt.show()

# X, y = final_bulk_reemst.drop(["Age"], axis=1), final_bulk_reemst.Age
# pca = PCA(n_components=2)
# #pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
# #plt.figure(figsize=(8,6))
# Xt = pca.fit_transform(X)
# df_temp = pd.DataFrame(dict(PC1 = Xt[:, 0], PC2 = Xt[:, 1], Age = y))
# colors_dict = {'P9': '#cc0000',
#                'P200': '#9933ff'
#                }
# sns.scatterplot(x=df_temp['PC1'], y=df_temp['PC2'], hue=df_temp['Age'], palette = colors_dict)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Mice bulk Data: 2D PCA")
# plt.show()


# PCA ON FETUS PSEUDOBULK
filtered_pseudobulk_fetus['Trimester'] = filtered_pseudobulk_fetus['Age'].apply(lambda x: 'Trimester 1' if x < 91 else 'Trimester 2')
# print(filtered_pseudobulk_fetus)
# X, y = filtered_pseudobulk_fetus.drop(["Age", 'Trimester'], axis=1), filtered_pseudobulk_fetus['Age']
# pca = PCA(n_components=2)
# Xt = pca.fit_transform(X)
# df_temp = pd.DataFrame(dict(PC1 = Xt[:, 0], PC2 = Xt[:, 1], Age = y))
# colors_dict = {'Trimester 1': '#cc0000',
#                'Trimester 2': '#9933ff'
#                }
# sns.scatterplot(x=df_temp['PC1'], y=df_temp['PC2'], hue= filtered_pseudobulk_fetus['Trimester'], palette = colors_dict)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Human Fetus Pseudobulk Data: 2D PCA")
# plt.show()




# # PART 2 (CLUSTER BY CLUSTER) - MICE DATA
# # for loop to loop through all 21 clusters
# age_mapping_1 = {'P9': 9, 'P200': 200}
# final_bulk_reemst['Age'] = final_bulk_reemst['Age'].map(age_mapping_1)
# for i in range(0, 21):
#     #only include columns for that particular cluster
#     filtered_columns = result_temp_m.filter(regex=f".*[^12]_{str(i)}$", axis=1)
#     filtered_pseudo_temp = filtered_columns.filter(regex=f"^(?:{'|'.join(common_elements_m)})_", axis=1)
#     filtered_pseudo_temp.columns = filtered_pseudo_temp.columns.str.split('_').str.get(0)
#     filtered_pseudo_temp = filtered_pseudo_temp[common_elements_m]
#     #standard scaler
#     scaler = StandardScaler()
#     scaler.fit(filtered_pseudo_temp)
#     scaled_data = scaler.transform(filtered_pseudo_temp)
#     filtered_pseudo = pd.DataFrame(scaled_data, columns = filtered_pseudo_temp.columns)
#     filtered_pseudo["Age"] = result_1_m.index.str.split('_').str[1]
#     pseudo_bulk_1 =  filtered_pseudo.drop(["Age"], axis=1)
#     pseudo_bulk_1.insert(0,"Age",filtered_pseudo.Age)
#     print(pseudo_bulk_1)

#     #Elastic Net between mice pseudobulk and bulk
#     age_mapping = {'E14': -7, 'P4': 4, 'P5': 5, 'P30': 30, 'P100': 100, 'Old': 540}
#     pseudo_bulk_1['Age'] = pseudo_bulk_1['Age'].map(age_mapping)
#     elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42) 
#     elastic_net.fit(pseudo_bulk_1.iloc[:, 1:], pseudo_bulk_1.iloc[:, 0])
#     print("hi",pseudo_bulk_1.iloc[:, 1:])
#     print("hi1",pseudo_bulk_1.iloc[:, 0])
#     predictions_all = elastic_net.predict(final_bulk_reemst.iloc[:, 1:]).round().astype(int)
#     print(predictions_all)
#     print(final_bulk_reemst.iloc[:, 0])

#     data = {}
#     for actual, predicted in zip(list(final_bulk_reemst.iloc[:, 0]), predictions_all):
#         if actual not in data:
#             data[actual] = []
#         data[actual].append(predicted)

#     data_lists = [data[age] for age in sorted(data.keys())]
#     positions = [1, 2]
#     plt.boxplot(data_lists, positions=positions)
#     plt.xticks(positions, ['9', '200'])
#     plt.yticks(range(min(predictions_all)-1, max(predictions_all)+1, 50))
#     plt.xlabel('Postnatal Age in Days (True)')
#     plt.ylabel('Postnatal Age in Days (Predicted)')
#     plt.title(f'Box Plot of True v.s. Predicted Age (Mice Bulk Dataset): Cluster {i}')
#     plt.show()#savefig("elastic_net_box_plot.png", format="png",dpi = 1200)

    

# #Elastic Net between mice pseudobulk and bulk (AVERAGED ACROSS CLUSTERS)
# age_mapping = {'E14': -7, 'P4': 4, 'P5': 5, 'P30': 30, 'P100': 100, 'Old': 540}
# filtered_pseudobulk_hammond['Age'] = filtered_pseudobulk_hammond['Age'].map(age_mapping)
# age_mapping_1 = {'P9': 9, 'P200': 200}
# final_bulk_reemst['Age'] = final_bulk_reemst['Age'].map(age_mapping_1)
# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42) 
# elastic_net.fit(filtered_pseudobulk_hammond.iloc[:, 1:], filtered_pseudobulk_hammond.iloc[:, 0])
# print("hi",filtered_pseudobulk_hammond.iloc[:, 1:])
# print("hi1",filtered_pseudobulk_hammond.iloc[:, 0])
# predictions_all = elastic_net.predict(final_bulk_reemst.iloc[:, 1:]).round().astype(int)
# print(predictions_all)
# print(final_bulk_reemst.iloc[:, 0])
# # print(accuracy(final_bulk_reemst.iloc[:, 0],predictions_all))

# data = {}
# for actual, predicted in zip(list(final_bulk_reemst.iloc[:, 0]), predictions_all):
#     if actual not in data:
#         data[actual] = []
#     data[actual].append(predicted)

# data_lists = [data[age] for age in sorted(data.keys())]
# positions = [1, 2]
# plt.boxplot(data_lists, positions=positions)
# plt.xticks(positions, ['9', '200'])
# plt.yticks(range(-200, max(predictions_all)+1, 50))
# plt.xlabel('Postnatal Age in Days (True)')
# plt.ylabel('Postnatal Age in Days (Predicted)')
# plt.title('Box Plot of True vs. Predicted Age (Mice Bulk Dataset)')
# plt.show()#savefig("elastic_net_box_plot.png", format="png",dpi = 1200)

# plt.clf()

# feature_importances = np.abs(elastic_net.coef_)
# top_10_genes_indices = np.argsort(feature_importances)[-10:]
# top_10_genes = filtered_pseudobulk_hammond.columns[top_10_genes_indices]
# pseudo_bulk_1 = filtered_pseudobulk_hammond.sort_values("Age")
# df_subset = pseudo_bulk_1[top_10_genes]

# plt.figure(figsize=(12, 6))
# sns.heatmap(df_subset, cmap='viridis', linewidths=0.5)
# plt.title('Heatmap of Expression Levels of Top 10 Most Important Genes (Mice Bulk Dataset)')
# plt.xlabel('Genes')
# plt.ylabel('Samples')
# plt.show()


# IGNORE THIS
# # Elastic Net between HUMAN pseudobulk and bulk
# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42) 
# filtered_pseudobulk_fetus = filtered_pseudobulk_fetus.drop(['Age'], axis = 1)
# print(filtered_pseudobulk_fetus)
# model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)
# model.fit(filtered_pseudobulk_fetus.iloc[:, :-1], filtered_pseudobulk_fetus['Trimester'])
# print(filtered_pseudobulk_fetus['Trimester'])
# # Predict
# y_pred = model.predict(final_bulk_fetus.iloc[:, 1:])
# # elastic_net.fit(filtered_pseudobulk_fetus.iloc[:, :-1], filtered_pseudobulk_fetus['Trimester'])
# # predictions_all = elastic_net.predict(final_bulk_fetus.iloc[:, 1:]).round().astype(int)
# print(y_pred)
# print(['Trimester 2','Trimester 2','Trimester 2','Trimester 2','Trimester 2','Trimester 2','Trimester 2','Trimester 2', 'Trimester 2'])#final_bulk_fetus.iloc[:, 0])



# Elastic Net between HUMAN pseudobulk and bulk
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42) 
filtered_pseudobulk_fetus = filtered_pseudobulk_fetus.drop(['Trimester'], axis = 1)
elastic_net.fit(filtered_pseudobulk_fetus.iloc[:, 1:], filtered_pseudobulk_fetus.iloc[:, 0])
predictions_all = elastic_net.predict(final_bulk_fetus.iloc[:, 1:]).round().astype(int)
print(predictions_all)
print(final_bulk_fetus.iloc[:, 0])

correlation_coefficient = np.corrcoef(list(final_bulk_fetus.iloc[:, 0]), predictions_all)[0, 1]
plt.figure(figsize=(8, 6))
plt.scatter(list(final_bulk_fetus.iloc[:, 0]), predictions_all, color='blue', alpha=0.5)  # Use alpha to control transparency
# plt.plot([min(list(final_bulk_fetus.iloc[:, 0])), max(list(final_bulk_fetus.iloc[:, 0]))], [min(list(final_bulk_fetus.iloc[:, 0])), max(list(final_bulk_fetus.iloc[:, 0]))], color='red')  # Add a diagonal line for reference
plt.xlabel('Gestational Age in Days (True)')
plt.ylabel('Gestational Age in Days (Predicted)')
plt.title('True v.s Predicted Age (Human Fetus Bulk Dataset)')
plt.text(min(list(final_bulk_fetus.iloc[:, 0])), max(predictions_all), f'Correlation Coefficient: {correlation_coefficient:.2f}', verticalalignment='top', horizontalalignment='left', color='black', fontsize=12)
plt.grid(True)
plt.show()

feature_importances = np.abs(elastic_net.coef_)
top_10_genes_indices = np.argsort(feature_importances)[-10:]
top_10_genes = filtered_pseudobulk_hammond.columns[top_10_genes_indices]
pseudo_bulk_1 = filtered_pseudobulk_hammond.sort_values("Age")
df_subset = pseudo_bulk_1[top_10_genes]

plt.figure(figsize=(12, 6))
sns.heatmap(df_subset, cmap='viridis', linewidths=0.5)
plt.title('Heatmap of Expression Levels of Top 10 Most Important Genes (Human Fetus Bulk Dataset)')
plt.xlabel('Genes')
plt.ylabel('Samples')
plt.show()




#PART 2 (CLUSTER BY CLUSTER) FOR HUMAN FETUS DATASET
#for loop to loop through all 16 clusters
for i in range(0, 16):
    #only include columns for that particular cluster
    filtered_columns = result_temp_h.filter(regex=f".*[^12]_{str(i)}$", axis=1)
    filtered_pseudo_temp = filtered_columns.filter(regex=f"^(?:{'|'.join(common_elements_h)})_", axis=1)
    filtered_pseudo_temp.columns = filtered_pseudo_temp.columns.str.split('_').str.get(0)
    filtered_pseudo_temp = filtered_pseudo_temp[common_elements_h]
    #standard scaler
    scaler = StandardScaler()
    scaler.fit(filtered_pseudo_temp)
    scaled_data = scaler.transform(filtered_pseudo_temp)
    filtered_pseudo = pd.DataFrame(scaled_data, columns = filtered_pseudo_temp.columns)
    filtered_pseudo['sample_name'] = result_temp_h.index.str.split('_').str[0]
    merged_df = pd.merge(filtered_pseudo, df_metadata, on='sample_name', how='left')
    merged_df['Gestational age (days)'] = merged_df['Gestational age (weeks + days)'].apply(convert_gest_age_to_days)

    pseudo_bulk_1 =  merged_df.drop(['Gestational age (days)', 'Gestational age (weeks + days)', 'Gestational week','sample_name'], axis=1)
    pseudo_bulk_1.insert(0,"Age",merged_df['Gestational age (days)'])
    # pseudo_bulk_1.to_csv("pseudo_bulk_1.csv")
    print(pseudo_bulk_1)
    #Elastic Net between HUMAN pseudobulk and bulk
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42) 
    elastic_net.fit(pseudo_bulk_1.iloc[:, 1:], pseudo_bulk_1.iloc[:, 0])
    predictions_all = elastic_net.predict(final_bulk_fetus.iloc[:, 1:]).round().astype(int)
    print(predictions_all)
    print(final_bulk_fetus.iloc[:, 0])

    correlation_coefficient = np.corrcoef(list(final_bulk_fetus.iloc[:, 0]), predictions_all)[0, 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(list(final_bulk_fetus.iloc[:, 0]), predictions_all, color='blue', alpha=0.5)  # Use alpha to control transparency
    #m, b = np.polyfit(predictions_all, list(final_bulk_fetus.iloc[:, 0]), 1)
    #plt.plot(np.array(predictions_all), m*np.array(predictions_all)+b,color = 'red')
    plt.xlabel('Gestational Age in Days (True)')
    plt.ylabel('Gestational Age in Days (Predicted)')
    plt.title(f'True vs. Predicted Age (Human Fetus Bulk Dataset): Cluster {i}')
    plt.text(min(list(final_bulk_fetus.iloc[:, 0])), max(predictions_all), f'Correlation Coefficient: {correlation_coefficient:.2f}', verticalalignment='top', horizontalalignment='left', color='black', fontsize=12)
    plt.grid(True)
    plt.show()



