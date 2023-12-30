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

#PCA
from community import community_louvain
import networkx as nx
import matplotlib.cm as cm

#gives you all brain modules
def compute_communities(dataframe, condition):

    results = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        G = row['graph']
        c_comp=nx.connected_components(G)

        com=[]
        for component in c_comp:
          component_subgraph = G.subgraph(component)
          com= com + greedy_modularity_communities(component_subgraph)
        #ls=community_metrics(G,com)
        #Put results in the result df
        results.append((row.name, com))
    return results

#computes closeness centrality and betweenness centrality fo each component
def community_metrics(G,com):
    ls=[]
    for i, comm in enumerate(com):
        subgraph = G.subgraph(comm)
        betweenness_centrality = centrality.betweenness_centrality(subgraph)
        closeness_centrality = centrality.closeness_centrality(subgraph)
        ls.append(betweenness_centrality,closeness_centrality)
    return ls
