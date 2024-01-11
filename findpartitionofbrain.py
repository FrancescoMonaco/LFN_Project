from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, ticker

from nilearn import datasets, plotting
from nilearn.image import get_data, index_img, mean_img
from nilearn.regions import Parcellations
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx

#gives you all brain modules and computes betweenness centrality and closeness centrality for each module
def compute_communities(dataframe, condition):

    results = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        G = row['graph']
        c_comp=nx.connected_components(G)

        com=[]
        #since the graph is not connected we compute the comunities for each connected
        #component in G
        for component in c_comp:
          component_subgraph = G.subgraph(component)
          com= com + greedy_modularity_communities(component_subgraph)
        
        #Put results in the result df
        results.append((row.name, com))
    return results
