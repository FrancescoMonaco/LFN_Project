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
        num_nodes=subgraph.number_of_nodes()
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
def modular_centrality(G):
    
    # Step 2: Remove all the inter-community links from the original network G.
    com = nx.connected_components(G)
    communities = []
    for component in com:
        component_subgraph = G.subgraph(component)
        communities = communities + list(greedy_modularity_communities(component_subgraph))
   
    local_networks = []
    for community in communities:
        local_network = G.subgraph(community)
        local_networks.append(local_network)

    # Step 3: Compute the Local measure for each node in its own community.
    local_measures = {}
    for local_network in local_networks:
        cc=nx.closeness_centrality(local_network)
        for node in local_network.nodes():
            local_measures[node] =cc[node] 
    print(local_measures)

    # Step 4: Remove all the intra-community links from the original network.
    inter_community_links = G.copy()
    
    for local_network in local_networks:
         
        inter_community_links.remove_edges_from(local_network.edges)

    # Step 5: Form the global network based on the union of all the connected components.
    global_network = nx.Graph()
    global_network.add_nodes_from(G.nodes())
    global_network.add_edges_from(inter_community_links.edges())

    # Step 6: Compute the Global measure of the nodes linking the communities.
    global_measures = nx.closeness_centrality(global_network)  # Use the chosen centrality measure on the global network
    print(global_measures)
    # Step 7: Add local and global measures to the Modular centrality vector.
    modular_centrality_vector = {}
    for node in G.nodes():
        modular_centrality_vector[node] = global_measures.get(node, 0) / local_measures.get(node, 0)
    
    top_5_nodes = sorted(modular_centrality_vector.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_modular_centrality = dict(top_5_nodes)
    print(top_5_modular_centrality)
    return modular_centrality_vector
