from graph_tool.all import *
from plotly.graph_objects import Figure, Scatter, Layout, Line, Annotation
import numpy as np
import networkx as nx
import pandas as pd

def compute_plot_top3_motifs(df, k_in=4):
    '''
    Compute and plot the top 3 motifs from a dataframe containing the graphs in the 'graph' column.

    Parameters:
    - df (pandas.DataFrame): Dataframe containing graphs in the 'graph' column.
    - k_in (int): Motif size to compute.

    Returns:
    Plotly figure.
    '''

    # Initialize a dictionary to store the total motif counts
    total_motif_counts = {}

    # Iterate over each row in control_data
    for index, row in df.iterrows():
        nx_graph = row['graph']

        # Transform networkX -> graph-tool
        g = Graph(directed=False)
        for edge in nx_graph.edges():
            e = g.add_edge(edge[0], edge[1])

        # Find the maximal connected component
        largest_comp = label_largest_component(g)
        subgraph = GraphView(g, vfilt=largest_comp)

        # Compute motif counts up to k=4
        motif_counts_tuple = motifs(subgraph, k=k_in)
        motif_graphs = motif_counts_tuple[0]
        motif_counts = motif_counts_tuple[1]

        # Sum the motif counts for each motif
        for motif_index, count in enumerate(motif_counts):
            if motif_index not in total_motif_counts:
                total_motif_counts[motif_index] = count
            else:
                total_motif_counts[motif_index] += count

    # Get the indices of the motifs sorted by total count in descending order
    sorted_motif_indices = np.argsort(list(total_motif_counts.values()))[::-1]

    # Extract the top 3 motifs
    top_motifs = sorted_motif_indices[:3]
    fig = Figure()

    # Iterate over the top motifs
    for motif_index in top_motifs:
        # Get the motif graph
        motif_graph = motif_graphs[motif_index]

        # Get positions for nodes
        pos = sfdp_layout(motif_graph)

        # Node coords
        Xn, Yn = zip(*pos)

        # Add the nodes
        node_trace = Scatter(x=Xn, y=Yn, mode='markers')
        fig.add_trace(node_trace)

        # Extract edges and their coordinates
        edges = motif_graph.edges()
        Xe, Ye = [], []
        for edge in edges:
            source, target = edge.source(), edge.target()
            Xe += [pos[source][0], pos[target][0], None]
            Ye += [pos[source][1], pos[target][1], None]

        # Add the edges
        edge_trace = Scatter(x=Xe, y=Ye, mode='lines')
        fig.add_trace(edge_trace)

        # Add their count
        motif_count_annotation = Annotation(
            text=f'Motif {motif_index + 1}<br>Count: {total_motif_counts[motif_index]}',
            x=pos[0][0],
            y=pos[0][1],
            xref='x',
            yref='y',
        )
        fig.add_annotation(motif_count_annotation)

    # Show the figure
    fig.show()

    return fig