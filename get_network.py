## From Section 4 notebook (Barry)

import pickle
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
import astropy.stats
from palettable import wesanderson
from sklearn.preprocessing import MultiLabelBinarizer
import urllib as ul
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx

df = pd.read_csv('smoking.tsv' , delimiter='\t') # Tab seperated document from EWAS


df_filt = df[['cpg' , 'gene']] # Subset to CpG and Genes
df_filt = df_filt[df_filt['gene'] != '-'] 

df_filt = df_filt.drop_duplicates()
df_filt = df_filt.set_index('cpg')

with open('/Users/chaeeunlee/Documents/VSC_workspaces/ismb_2024_data/ISMB_GS_Data.pkl' , 'rb') as file : 
    loaded_data = pd.read_pickle(file)
    
w1 = loaded_data['DNAm_w1'].dropna()
w3 = loaded_data['DNAm_w3'].dropna()
phenotypes = loaded_data['Phenotypes']

common_cpg = list(set(w1.index) & set(w3.index) & set(df_filt.index)) # Identify CpG sites present in both W1 and W3 cohorts
w1 = w1.loc[common_cpg]
w3 = w3.loc[common_cpg]
df_filt = df_filt.loc[common_cpg]

def smoking_cat(pack_years) : 
    if pack_years == 0 :
        return 'Never'
    #elif pack_years < 10 : 
    #    return 'Infrequent'
    else :
        return 'Smoker'

phenotypes['Smoking'] = phenotypes['pack_years'].apply(smoking_cat) 

phenotypes_w1 = phenotypes.set_index('Sample_SentrixID')
phenotypes_w1 = phenotypes_w1[phenotypes_w1['Smoking'].notna()]
phenotypes_w1 = phenotypes_w1[phenotypes_w1['Set'] == 'W1']

common_idx = list(set(w1.columns) & set(phenotypes_w1.index))
w1 = w1.loc[: , common_idx]
phenotypes_w1 = phenotypes_w1.loc[common_idx]

phenotypes_w3 = phenotypes.set_index('Sample_SentrixID')
phenotypes_w3 = phenotypes_w3[phenotypes_w3['Smoking'].notna()]
phenotypes_w3 = phenotypes_w3[phenotypes_w3['Set'] == 'W3']

common_idx = list(set(w3.columns) & set(phenotypes_w3.index))
w3 = w3.loc[: , common_idx]
phenotypes_w3 = phenotypes_w3.loc[common_idx]


external_gene_names = '%0D'.join(map(str,df_filt['gene'].iloc[:300].drop_duplicates())) # all of em are too long

query_url = 'https://string-db.org/api/tsv-no-header/get_string_ids?identifiers='+external_gene_names+'&species=9606&format=only-ids'

# use the urllib library to retrieve the String-DB internal IDs
result = ul.request.urlopen(query_url).read().decode('utf-8')

# now we want to query String-DB to retrieve interactions from this list of String-DB IDs
# we create a concatenated list of stringdbIDs in much the same way as above for the Entrez Gene IDs
stringdbIDs = '%0D'.join(result.splitlines())

# again we build the query for interactions using the String-DB IDs
query_url = 'https://string-db.org/api/tsv/network?identifiers='+stringdbIDs+'&species=9606'

# again using urllib to retrieve the interactions these are returned in a standard tab delimied text format
interactions = ul.request.urlopen(query_url).read().decode('utf-8').splitlines()

# we need to split the result by these 'tabs' (\t - is used to identfy them)
int_test = [interaction.split('\t') for interaction in interactions]

# we extract the field names from the first row
column_names = int_test[:1][0]

# create a Pandas dataframe of the interaction data we have just retrieved from String-DB
interactions_df = pd.DataFrame(int_test,columns=column_names)

# delete the first row that held the fieldnames but we no longer need
interactions_df = interactions_df.drop(labels=0,axis=0)

# remove any duplicate rows
final_interactions = interactions_df.drop_duplicates()


row_keep = []
for i , row in final_interactions.iterrows() : 
    if (row['preferredName_A'] in list(df_filt['gene'])) & (row['preferredName_B'] in list(df_filt['gene']))  : 
        row_keep.append(i)
        
final_interactions = final_interactions.loc[row_keep]

## Get network

 #Create an empty graph
G = nx.Graph()

# add all nodes
G.add_nodes_from(set(final_interactions['preferredName_A']) | set(final_interactions['preferredName_B'])) 

# add the edges (connections) to the network 
edges = []
for edge1 , edge2  in zip(final_interactions['preferredName_A'] , final_interactions['preferredName_B']) : #add all edge to the network
    edges.append((edge1 , edge2 ))
G.add_edges_from(edges)

### from further down Barry's code

gene_network_subset = df_filt[df_filt['gene'].isin(list(G.nodes))]

w1_net = w1.loc[gene_network_subset.index]
w1_net.index = gene_network_subset['gene']
w1_net = w1_net.reset_index().groupby('gene').mean()

w3_net = w3.loc[gene_network_subset.index]
w3_net.index = gene_network_subset['gene']
w3_net = w3_net.reset_index().groupby('gene').mean()

# from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

w1_labels = mlb.fit_transform(phenotypes_w1['Smoking'].values.reshape(-1,1))
w3_labels = mlb.transform(phenotypes_w3['Smoking'].values.reshape(-1,1))