import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random


def graph_plot(G):
    '''
    Plot a graph using plotly.

    Parameters:
    - G (networkx.Graph): Input graph.

    Returns:
    Plotly figure.
    '''
    pos = nx.spring_layout(G)
    node_degrees = dict(G.degree())

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

    # Show the interactive plot
    fig.show()

    return fig

def community_plot(G, node_community):
    '''
    Plot the graph color-coding the nodes based on their community.

    Parameters:
    - G (networkx.Graph): Input graph.
    - node_community (list): List of community assignments for each node.

    Returns:
    Plotly figure.

    '''
    node_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(G.nodes()))]

    # Map community assignments to corresponding colors
    node_colors = [node_colors[comm] for comm in node_community]

    scatter = go.Scatter(
        x=[pos[0] for pos in nx.spring_layout(G).values()],
        y=[pos[1] for pos in nx.spring_layout(G).values()],
        mode='markers',
        marker=dict(
            size=10,
            color=node_colors,  # Use random colors for nodes
            colorbar=None
        )
    )

    # Create the layout
    layout = go.Layout(
        title='Community Detection using Greedy Modularity',
        xaxis=dict(title=''),
        yaxis=dict(title='')
    )

    # Create the figure
    fig = go.Figure(data=[scatter], layout=layout)

    # Show the figure
    fig.show()

    return fig