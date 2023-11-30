#@title Functions to process the graphs
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from tabulate import tabulate


# Centralities
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

    return (m_closeness, m_betweenness, m_degree, avg_clust,
                    values_top_nodes_closeness,
                    values_top_nodes_betweenness,
                    values_top_nodes_degree,
                    values_top_nodes_clustering)

# Efficiency metrics
def global_brain_efficiency(G):
  return nx.global_efficiency(G)

def network_cost(G):
  n_edges=G.number_of_edges()
  n_nodes=G.number_of_nodes()
  binomial_coefficient = (stats.binom(n_nodes, 2)).pmf(2)
  return n_edges/binomial_coefficient

def process_graph_modularity(G):
  modularity = nx.community.modularity(G, nx.community.label_propagation_communities(G))
  return modularity

def process_graphs(dataframe, condition):
    '''
    Process the graphs in the dataframe and return a dataframe with the results

    Parameters:
    - dataframe (pandas.DataFrame): Input dataframe.
    - condition (str): Condition of the dataframe.

    Returns:
    Pandas dataframe containing the results.
    '''

    results = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        G = row['graph']

        # Mean centralities and top 5 values
        (m_closeness, m_betweenness, m_degree, avg_clust,
        values_top_nodes_closeness, values_top_nodes_betweenness,
        values_top_nodes_degree, values_top_nodes_clustering,
        ) = process_graph_centralities(G)

        # Modularity, costs
        modularity = process_graph_modularity(G)
        gbe = global_brain_efficiency(G)
        nc = network_cost(G) #launches an error, needs to be checked

        #Put results in the result df
        results.append((row.name, m_closeness, m_betweenness, m_degree, avg_clust,
                        values_top_nodes_closeness, values_top_nodes_betweenness,
                        values_top_nodes_degree, values_top_nodes_clustering,
                        modularity, gbe, nc))
    return results

def print_mean_std(dataframes, conditions, metrics=[
    'closeness', 'betweenness', 'degree', 'clustering',
    'top5_close', 'top5_betw', 'top5_deg', 'top5_clust',
    'modularity', 'global_efficiency', 'network_cost', 'edges', 'nodes'
]):
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