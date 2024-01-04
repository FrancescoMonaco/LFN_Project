#import time

from scipy.integrate import simpson
from numpy import trapz
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, ticker

from nilearn import datasets, plotting
from nilearn.image import get_data, index_img, mean_img
from nilearn.regions import Parcellations

import matplotlib.pyplot as plt
from nilearn import plotting
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
#PCA
from community import community_louvain
import networkx as nx
import matplotlib.cm as cm

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
        lc,lb=community_metrics(G,com)
        #Put results in the result df
        results.append((row.name, com,lc,lb))
    return results

#computes closeness centrality and betweenness centrality fo each module
def community_metrics(G,com):
    lc=[]
    lb=[]
    #take all the modules in com
    for i, comm in enumerate(com):
        subgraph = G.subgraph(comm)
        betweenness = nx.betweenness_centrality(subgraph)
        closeness = nx.closeness_centrality(subgraph)
        # Sort nodes based on centrality measures
        top_nodes_closeness = sorted(closeness, key=closeness.get, reverse=True)[:5]
        top_nodes_betweenness = sorted(betweenness, key=betweenness.get, reverse=True)[:5]
        # Extract their values
        values_top_nodes_closeness = np.mean([closeness[node] for node in top_nodes_closeness])
        values_top_nodes_betweenness = np.mean([betweenness[node] for node in top_nodes_betweenness])
        m_closeness = sum(closeness.values())/num_nodes
        m_betweenness = sum(betweenness.values())/num_nodes
    
        lc.append(m_closeness)
        lb.append(m_betweenness)
        
    return lc,lb
