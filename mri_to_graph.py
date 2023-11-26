#@title Processing functions
#*** To parallelize the computations of the graphs
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker, NiftiSpheresMasker
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def process_dataframe_row(data_path, confounds_data, spheres_masker, threshold):
    confounds_data = confounds_data.replace([np.nan, np.inf, -np.inf], 0)
    timeseries = spheres_masker.fit_transform(data_path, confounds=confounds_data)

    # Scale the data
    scaler = StandardScaler()
    timeseries_normalized = scaler.fit_transform(timeseries)

    power_correlation_matrix = np.corrcoef(timeseries_normalized, rowvar=False)

    # Create the graph
    G = nx.Graph()
    n = power_correlation_matrix.shape[0]  # Number of brain regions
    # Don't count the self correlation
    mask = ~np.eye(n, dtype=bool)

    # Calculate the threshold based on the top 0.n% values
    values = power_correlation_matrix[mask].flatten()

    # Sort the values
    sorted_values = np.sort(values)[::-1]

    # Calculate the threshold
    top_percentage = threshold
    threshold_idx = int(len(sorted_values) * top_percentage / 100)
    threshold = sorted_values[threshold_idx]

    for i in range(len(power_correlation_matrix)):
        for j in range(i + 1, len(power_correlation_matrix)):
            if power_correlation_matrix[i, j] >= threshold:
                G.add_edge(i, j)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    return G, num_nodes, num_edges

def process_dataframe(dataframe, condition, spheres_masker, threshold):
    results = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        data_path = row['data']
        confounds_data = row['confounds']
        G, num_nodes, num_edges = process_dataframe_row(data_path, confounds_data, spheres_masker, threshold)
        results.append((row.name, G, num_nodes, num_edges))
    return results