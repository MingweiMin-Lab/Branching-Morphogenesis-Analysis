import h5py
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Union
from IPython.display import IFrame, display
import plotly.io as pio
from IPython import get_ipython
import sys
import matplotlib.patches as mpatches
from sklearn.datasets import make_blobs, make_swiss_roll
import umap
sys.path.append("./functions")  
import powersmooth2 as ps
import math_calculation as ac

pio.renderers.default = 'iframe'  # or try'jupyterlab'


def increase_data_rate():
    ipython = get_ipython()
    if ipython is not None:
        ipython.config.NotebookApp.iopub_data_rate_limit = 1.0e8
        print("Data rate limit increased within the notebook.")
    else:
        print("Could not access IPython kernel.")


def visualize_tree(h5_filepath: str, output_filepath: Optional[str] = None, use_plotly_if_large: bool = True,
                   edge_label_type: str = 'length', node_label_type: str = 'id', show_node_labels=False,
                   display_inline: bool = True, max_depth: Optional[int] = None, pos: Optional[int] = None):
    graph = get_graph(h5_filepath)

    if use_plotly_if_large and num_nodes > 200:
        _visualize_tree_plotly(graph, output_filepath, edge_label_type, node_label_type, show_node_labels,
                               display_inline)
    else:
        visualize_tree_matplotlib(graph, output_filepath, edge_label_type, max_depth, pos)



def plot_edge(G, parent, node, elev, azim):
    vecIn = G[parent][node]['vecIn']
    vecOut = G[parent][node]['vecOut']
    edgeX = G[parent][node]['x']
    edgeY = G[parent][node]['y']
    edgeZ = G[parent][node]['z']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.plot3D(edgeX, edgeY, edgeZ, color='maroon') 
    ax.quiver(edgeX[0], edgeY[0], edgeZ[0], vecIn[0], vecIn[1], vecIn[2], color='green', length=3,
              arrow_length_ratio=0.3)
    ax.quiver(edgeX[-1], edgeY[-1], edgeZ[-1], vecOut[0], vecOut[1], vecOut[2], color='blue', length=3,
              arrow_length_ratio=0.3)
    ax.view_init(elev=elev, azim=azim)  
    plt.show()
    return ax


def plot_edge_2D(G, parent, node):
    vecIn = G[parent][node]['vecIn']
    vecOut = G[parent][node]['vecOut']
    edgeX = G[parent][node]['x']
    edgeY = G[parent][node]['y']
    edgeZ = G[parent][node]['z']

    vec_len = 3
    plt.subplot(1, 2, 1)
    plt.plot(edgeX, edgeY, color='red')
    plt.arrow(edgeX[0], edgeY[0], vecIn[0] * vec_len, vecIn[1] * vec_len,
              width=0.3, head_width=0.3,
              head_length=0.3, color='green')
    plt.arrow(edgeX[-1], edgeY[-1], vecOut[0] * vec_len, vecOut[1] * vec_len,
              width=0.3, head_width=0.3,
              head_length=0.3, color='blue')
    plt.subplot(1, 2, 2)

    plt.plot(edgeX, edgeZ, color='red')
    plt.arrow(edgeX[0], edgeZ[0], vecIn[0] * vec_len, vecIn[2] * vec_len,
              width=0.3, head_width=1,
              head_length=1, color='green')
    plt.arrow(edgeX[-1], edgeZ[-1], vecOut[0] * vec_len, vecOut[2] * vec_len,
              width=0.3, head_width=1,
              head_length=1, color='blue')
    plt.show()


def visualize_tree_matplotlib(graph, exported_figure_path=None, edge_label_type=None, max_depth=None, pos=None, color_data=None, node_size = 100, font_size = 10):

    plt.figure(figsize=(8, 8))

    if max_depth is not None:
        subgraph = extract_subtree_bfs(graph, root=0, max_depth=max_depth)
    else:
        subgraph = graph
    if pos is None:
        pos = _hierarchy_pos(subgraph, root=0)
    # xList = nx.get_node_attributes(graph, 'x')
    # yList = nx.get_node_attributes(graph, 'y')
    # zList = nx.get_node_attributes(graph, 'z')
    # pos = {k: (v1, v2) for (k, v1), (_, v2) in zip(xList.items(), zList.items())}


    # degrees = nx.get_node_attributes(graph, 'nChild')
    out_degrees = dict(graph.out_degree())
    c_data = list(out_degrees.values()) # degrees
    if color_data is not None:
        c_data = color_data

    norm = plt.Normalize(vmin=min(c_data), vmax=max(c_data))
    # cmap = plt.cm.viridis  
    cmap = plt.cm.spring
    nx.draw(subgraph, pos, node_color=[cmap(norm(deg)) for deg in c_data], with_labels=True, node_size=node_size,
            font_size=font_size, width=1.0)

    if edge_label_type is not None:
        if edge_label_type == 'length':
            edge_labels = {(u, v): f"{subgraph[u][v]['length']:.2f}" for u, v in subgraph.edges()}
        elif edge_label_type == 'radius_avg':
            edge_labels = {(u, v): f"{subgraph[u][v]['radius_avg']:.2f}" for u, v in subgraph.edges()}
        else:
            raise TypeError("edge_label_type must be 'length' or 'radius_avg'")
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

    if exported_figure_path:
        plt.savefig(exported_figure_path, format="pdf", bbox_inches="tight", transparent=True)
    else:
        plt.axis('equal')
        plt.show()


def _hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    # (same as before)
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = np.random.choice(list(G.nodes))

    def _hierarchy_pos_recursive(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos_recursive(G, child, width=dx, vert_gap=vert_gap,
                                               vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                               pos=pos, parent=root)
        return pos

    return _hierarchy_pos_recursive(G, root, width, vert_gap, vert_loc, xcenter)

    import networkx as nx


def plot_graph_3D(G):
    edges = list(G.edges(data=True))  

    x_edges, y_edges, z_edges = [], [], []
    for u, v, attr in G.edges(data=True):
        x_u, y_u, z_u = G.nodes[u]['x'], G.nodes[u]['y'], G.nodes[u]['z']
        x_v, y_v, z_v = G.nodes[v]['x'], G.nodes[v]['y'], G.nodes[v]['z']
        x_edges.extend([x_u, x_v, None])
        y_edges.extend([y_u, y_v, None])
        z_edges.extend([z_u, z_v, None])

    edge_trace = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(width=2, color='gray')  
    )

    node_x = [attr['x'] for _, attr in G.nodes(data=True)]
    node_y = [attr['y'] for _, attr in G.nodes(data=True)]
    node_z = [attr['z'] for _, attr in G.nodes(data=True)]
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers+text',
        marker=dict(size=3, color='blue'),
        text=list(G.nodes()),
        textposition='top center'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        width=1000,   # in pixel
        height=800,   # in pixel
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(aspectmode='cube')
    )
    # fig.update_layout(scene=dict(aspectmode='cube'))
    fig.show()


def scatter_with_label(xy_data, label, label_names, name_list, c_list):
    categories = np.unique(label)
    fig = go.Figure()

    for cat in categories:
        mask = (label == cat)
        fig.add_trace(go.Scatter(
            x=xy_data[mask, 0],
            y=xy_data[mask, 1],
            mode='markers',
            name=label_names[cat],
            marker=dict(color=c_list[cat]),
            text = name_list[mask],  
            hoverinfo='text',
            showlegend= True
        ))

    fig.update_layout(
        legend=dict(orientation='v', y=1.02),
        title='UMAP Projection of Branching patterns'
    )
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.show()


def plot_umap1(title, X_2d, labels, label_names, c_list, xlabel='UMAP1', ylabel='UMAP2', figure_name = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    # scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=[c_list[i] for i in labels], s=10, alpha=0.8)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=c_list[labels], s=10, alpha=0.8)
    legend_handles = [mpatches.Patch(color=c_list[i], label=label_names[i]) for i in range(0, len(label_names))]
    plt.legend(handles=legend_handles)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)    
    if (figure_name is not None):
        plt.savefig(figure_name, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()
    return fig, ax
    
def plot_umap2(title, X_2d, labels, label_names, cmap, xlabel='UMAP1', ylabel='UMAP2'):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.8)
    plt.colorbar(scatter, label='Labels')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def main():
    increase_data_rate()  # Call to set the limit


if __name__ == "__main__":
    main()
