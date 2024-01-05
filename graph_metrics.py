#@title Functions to process the graphs
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from tabulate import tabulate
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities

# Metrics to be calculated
base_metrics = [
    'closeness', 'betweenness', 'degree', 'clustering',
    'top5_close', 'top5_betw', 'top5_deg', 'top5_clust',
    'modularity', 'global_efficiency', 'edges', 'nodes',
    'assortativity', 'transitivity','local_efficiency','top_nodes_modular_closeness']

# ***Features
def process_graph_centralities(G):
    """
    Calculate centrality measures for a graph and extract top 5 nodes for each measure.

    Parameters:
    - G (networkx.Graph): Input graph.

    Returns:
    Tuple containing:
    1. Mean closeness centrality of the entire graph.
    2. Mean betweenness centrality of the entire graph.
    3. Mean degree centrality of the entire graph.
    4. Average clustering coefficient of the entire graph.
    5. List of values for the top 5 nodes based on closeness centrality.
    6. List of values for the top 5 nodes based on betweenness centrality.
    7. List of values for the top 5 nodes based on degree centrality.
    8. List of values for the top 5 nodes based on clustering coefficient.
    """

    num_nodes = G.number_of_nodes()

    closeness = nx.closeness_centrality(G)

    betweenness = nx.betweenness_centrality(G)

    degree = dict(G.degree())

    clustering = nx.clustering(G)
    
    # Sort nodes based on centrality measures
    top_nodes_closeness = sorted(closeness, key=closeness.get, reverse=True)[:5]
    top_nodes_betweenness = sorted(betweenness, key=betweenness.get, reverse=True)[:5]
    top_nodes_degree = sorted(degree, key=degree.get, reverse=True)[:5]
    top_nodes_clustering = sorted(clustering, key=clustering.get, reverse=True)[:5]
    
    # Extract their values
    values_top_nodes_closeness = np.mean([closeness[node] for node in top_nodes_closeness])
    values_top_nodes_betweenness = np.mean([betweenness[node] for node in top_nodes_betweenness])
    values_top_nodes_degree = np.mean([degree[node] for node in top_nodes_degree])
    values_top_nodes_clustering = np.mean([clustering[node] for node in top_nodes_clustering])
    
    # Mean on the graph
    m_degree = sum(degree.values())/num_nodes
    m_closeness = sum(closeness.values())/num_nodes
    m_betweenness = sum(betweenness.values())/num_nodes
    avg_clust = nx.average_clustering(G)
    #m_eccencitricy=

    return (m_closeness, m_betweenness, m_degree, avg_clust,
                    values_top_nodes_closeness,
                    values_top_nodes_betweenness,
                    values_top_nodes_degree,
                    values_top_nodes_clustering)

# Efficiency metrics
def global_brain_efficiency(G):
  return nx.global_efficiency(G)
    
def average_local_efficience(G):
    return nx.local_efficiency(G)
    
#modularity
def process_graph_modularity(G):
  components=nx.connected_components(G)
  modularity=0
  for x in components:
      modularity = modularity+nx.community.modularity(G, greedy_modularity_communities(G))
  return modularity

#assortativity 
def assortative(G):
  r= nx.degree_pearson_correlation_coefficient(G)
  return r

#transitivity
def transitive(G):
  return nx.transitivity(G)

#module closeness centrality
def modular_closeness_centrality(G):
    
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
    #print(local_measures)

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
    #print(global_measures)
    # Step 7: Add local and global measures to the Modular centrality vector.
    modular_centrality_vector = {}
    for node in G.nodes():
        if local_measures.get(node,0) != 0:
          modular_centrality_vector[node] = global_measures.get(node, 0) / local_measures.get(node, 0)
         else:
          modular_centrality_vector[node]=0
    
    top_5_nodes = sorted(modular_centrality_vector.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_modular_centrality = dict(top_5_nodes)
    print(top_5_modular_centrality)
    return top_5_modular_centrality
    
# ***Processing
def process_graphs(dataframe, condition):
    '''
    Process the graphs in the dataframe and return a dataframe with the results

    Parameters:
    1. dataframe (pandas.DataFrame): Input dataframe.
    2. condition (str): Condition of the dataframe.

    Returns:
    Pandas dataframe containing the results.
    '''

    results = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        G = row['graph']

        # Mean centralities and top 5 values
        
        (m_closeness, m_betweenness, m_degree, avg_clust,
        values_top_nodes_closeness, values_top_nodes_betweenness,
        values_top_nodes_degree, values_top_nodes_clustering
        ) = process_graph_centralities(G)

        # Modularity, costs
        modularity = process_graph_modularity(G)
        
        gbe = global_brain_efficiency(G)
        
        lc= average_local_efficience(G)
        
        #assortativity of the brain
        ass=assortative(G)
        
        #transitivity of the brain
        tran=transitive(G)

        top_nodes_modular_closeness=modular_closeness_centrality(G)
        
        #Put results in the result df
        results.append((row.name, m_closeness, m_betweenness, m_degree, avg_clust,
                        values_top_nodes_closeness, values_top_nodes_betweenness,
                        values_top_nodes_degree, values_top_nodes_clustering,
                        modularity, gbe,ass,tran,lc,top_nodes_modular_closeness))
    return results

def print_mean_std(dataframes, conditions, metrics=base_metrics):
  '''
  Print a table with mean and std for each condition and metric

  Parameters:
  1. dataframes (list): List of dataframes to be processed.
  2. conditions (list): List of conditions.
  3. metrics (list): List of metrics to be calculated.
  '''
  #print a table with mean and std for each condition and metric
  table_data = []

  for column in metrics:
      row = [column]
      for condition, df in zip(conditions, dataframes):
          mean_value = df[column].mean()
          std_value = df[column].std()
          row.extend([f"{mean_value:.4f} ± {std_value:.4f}"])
      table_data.append(row)

  headers = ["Metric"] + conditions
  table = tabulate(table_data, headers, tablefmt="pretty")
  print(table)
  return

def print_ttest_pval(dataframes, conditions, metrics=[
    'closeness', 'betweenness', 'degree', 'clustering',
    'top5_close', 'top5_betw', 'top5_deg', 'top5_clust',
    'modularity', 'global_efficiency']):
  '''
  Print a table with the p-value of the t-test for each condition and metric

  Parameters:
  1. dataframes (list): List of dataframes to be processed.
  2. conditions (list): List of conditions.
  3. metrics (list): List of metrics to be calculated.
  '''
  table_data = []

  for column in metrics:

    # Extract the control, we will compare the other conditions to it
    control_data = dataframes[0][column]
    row = [column]

    for condition, df in  zip(conditions[1:4], dataframes[1:4]):
      # Extract the data
      data = df[column]
      # Perform the t-test
      ttest = stats.ttest_ind(control_data, data)
      # Add the p-value to the row
      row.append(f"{ttest.pvalue:.4f}")
    table_data.append(row)

  headers = ["Metric"] + conditions[1:]
  table = tabulate(table_data, headers, tablefmt="pretty")
  print(table)
  return
