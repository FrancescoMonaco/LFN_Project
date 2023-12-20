import time

from scipy.integrate import simpson
from numpy import trapz

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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def decomposition(correlation_matrix):
  # Create a PCA instance
  pca = PCA()
  pca.fit(correlation_matrix)

  # Get the eigenvalues (principal components)
  eigenvectors = pca.components_

  # Plot the eigenvectors as arrows
  plt.figure(figsize=(8, 6))

  for i, (eigenvector, explained_variance_ratio) in enumerate(zip(eigenvectors, pca.explained_variance_ratio_)):
      plt.bar(range(len(eigenvector)), eigenvector, alpha=0.7, label=f'PC{i + 1}')

  plt.xlabel('Feature Number')
  plt.ylabel('Principal Component Value')
  plt.title('Principal Components of Correlation Matrix')
  plt.show()


def louvain_partitioning(control_data):
  G=control_data.at[9, 'graph']
  partition = community_louvain.best_partition(G,weight='1', resolution= 0.9)

  print(max(partition.values()))

  # draw the graph
  pos = nx.spring_layout(G)  # You can experiment with other layout algorithms
  # color the nodes according to their partition
  cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
  nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
  nx.draw_networkx_edges(G, pos, alpha=0.5)
  plt.show()





  # Draw the Louvain communities
  plt.figure(figsize=(12, 6))




  #da riadattare

  # Create a Plotly figure
  fig = go.Figure()

  # Add nodes to the figure with size proportional to degree
  for node, (x, y) in pos.items():
      fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=1 * node_degrees[node], opacity=0.7), name=str(node_degrees[node])))

  # Add edges to the figure
  edge_x = []
  edge_y = []
  for edge in G.edges():
      x0, y0 = pos[edge[0]]
      x1, y1 = pos[edge[1]]
      edge_x.extend([x0, x1, None])
      edge_y.extend([y0, y1, None])
  fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5)))





  # Subplot for Louvain communities
  plt.subplot(1, 2, 1)
  pos = nx.spring_layout(G)
  cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
  nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
  nx.draw_networkx_edges(G, pos, alpha=0.5)
  plt.title("Louvain Communities")

  # Subplot for connectome plot
  plt.subplot(1, 2, 2)
  # plot in nilearn the nodes whose color is assigned according to their partition
  display = plotting.plot_connectome(G, edge_cmap='viridis', node_color=list(partition.values()),
                          edge_vmin=0., edge_vmax=1., node_size=20, colorbar=False,
                          title="Connectome plot with communities")


  plt.show()


#gives you all brain modules
  def compute_communities(G):
    modules = nx.greedy_modularity_communities(G)
    return modules
