import h5py
import networkx as nx
from collections import deque
import numpy as np


def get_info(h5_filepath: str):
    with h5py.File(h5_filepath, 'r') as hf:
        info_group = hf['info']
        info_dict = {
            name: info_group[name][()]
            for name in info_group.keys()
            if isinstance(info_group[name], h5py.Dataset)
        }
    return info_dict


def write_info(h5_filepath: str, info_dict):
    with h5py.File(h5_filepath, "a") as hf:
        if hf.get('info') is None:
            info_group = hf.create_group("/info")
        else:
            info_group = hf['info']
        for key, value in info_dict.items():
            existing_ds = info_group.get(key)
            if existing_ds is None:
                info_group.create_dataset(key, data=value)
            else:
                info_group[key][()] = value


def get_graph(h5_filepath: str):

    with h5py.File(h5_filepath, 'r') as hf:
        if 'vertices' not in hf or 'edges' not in hf:
            raise ValueError("HDF5 file must contain 'vertices' and 'edges' groups.")

        graph = nx.DiGraph()
        num_nodes = 0

        for vertex_name in hf['vertices']:
            num_nodes += 1
            vertex_group = hf['vertices'][vertex_name]
            node_id = int(vertex_name[1:])
            xyz = vertex_group['xyz'][()]
            nChild = vertex_group['nChild'][()]
            parent_id = vertex_group['parent'][()]
            graph.add_node(node_id, parent=parent_id, nChild=nChild, x=xyz[0], y=xyz[1], z=xyz[2])

            if parent_id != -1:
                graph.add_edge(parent_id, node_id)
                parent_group = hf['vertices'][f'v{parent_id}']
                for i in range(1, parent_group['nChild'][()] + 1):
                    if parent_group[f'child{i}'][()] == node_id:
                        edge_id = parent_group[f'edge{i}'][()]
                        vecIn = parent_group[f'vecOut{i}'][()]
                        break
                else:
                    raise ValueError(f"Could not find edge from parent {parent_id} to child {node_id}")
                edge_group = hf['edges'][f'e{edge_id}']
                graph.edges[parent_id, node_id]['length'] = edge_group['length'][()]
                graph.edges[parent_id, node_id]['radius_avg'] = edge_group['radius_avg'][()]
                graph.edges[parent_id, node_id]['edge_id'] = edge_id
                graph.edges[parent_id, node_id]['vecIn'] = vecIn
                graph.edges[parent_id, node_id]['vecOut'] = vertex_group['vecIn'][()]
                if 'x' in edge_group:
                    edge_x = edge_group['x'][()]
                    edge_y = edge_group['y'][()]
                    edge_z = edge_group['z'][()]
                    graph.edges[parent_id, node_id]['x'] = edge_x
                    graph.edges[parent_id, node_id]['y'] = edge_y
                    graph.edges[parent_id, node_id]['z'] = edge_z
                else:
                    xyz_parent = parent_group['xyz'][()]
                    graph.edges[parent_id, node_id]['x'] = [xyz_parent[0], xyz[0]]
                    graph.edges[parent_id, node_id]['y'] = [xyz_parent[1], xyz[1]]
                    graph.edges[parent_id, node_id]['z'] = [xyz_parent[2], xyz[2]]
    return graph


def extract_subtree_bfs(graph, root=0, max_depth=3, include_parent = False):
    
    nodes_within_depth = {root}
    queue = deque([(root, 0)])

    while queue:
        current_node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in graph.neighbors(current_node):
            if neighbor not in nodes_within_depth:
                nodes_within_depth.add(neighbor)
                queue.append((neighbor, depth + 1))
    if include_parent and root != 0:
        parent = graph.nodes[root]['parent']
        nodes_within_depth.add(parent)

    subgraph = nx.DiGraph(graph.subgraph(nodes_within_depth))
    subgraph = update_parent_nChild(subgraph)
        
    return subgraph


def extract_subtree(graph, root=0, max_depth=3):
    nodes_within_depth = {root}
    for d in range(max_depth + 1):
        nodes_within_depth.update(nx.descendants_at_distance(graph, 0, d))
    return graph.subgraph(nodes_within_depth)


def renumber_nodes_edges(graph: nx.DiGraph) -> nx.DiGraph:
    # === renumber nodes ===
    # renumber using BFS（keep topological structure）
    root = next(n for n, data in graph.nodes(data=True) if data.get('parent') == -1)
    queue = deque([root])
    node_id_map = {}
    new_node_id = 0

    while queue:  # BFS
        current = queue.popleft()
        node_id_map[current] = new_node_id
        new_node_id += 1
        queue.extend(sorted(graph.successors(current)))

    # === renumber nodes ===
    edge_id_map = {}
    new_edge_id = 0
    for u, v in graph.edges():  # Traverse the original edges
        original_id = graph.edges[u, v]['edge_id']
        if original_id not in edge_id_map:
            edge_id_map[original_id] = new_edge_id
            new_edge_id += 1

    # === create a new graph ===
    new_graph = nx.DiGraph()

    # copy node attributes（update parent node ID）
    for old_id in graph.nodes():
        data = graph.nodes[old_id].copy()
        if data['parent'] != -1:
            data['parent'] = node_id_map[data['parent']]  # updating parent node ID
        new_graph.add_node(node_id_map[old_id], **data)

    # copy edge attributes（renumber edge ID）
    for u, v in graph.edges():
        edge_data = graph.edges[u, v].copy()
        edge_data['edge_id'] = edge_id_map[edge_data['edge_id']]  # update edge_id
        new_graph.add_edge(node_id_map[u], node_id_map[v], **edge_data)

    return new_graph


def save_graph(h5_filepath: str, graph: nx.DiGraph):
    """write graph into a HDF5 file"""
    with h5py.File(h5_filepath, 'w') as hf:
        # write nodes
        vertices = hf.create_group("vertices")
        for node_id in graph.nodes():
            node = graph.nodes[node_id]
            v_group = vertices.create_group(f"v{node_id}")
            v_group.create_dataset("xyz", data=[node['x'], node['y'], node['z']])
            v_group.create_dataset("nChild", data=node['nChild'])
            v_group.create_dataset("parent", data=node['parent'])
            parent = node['parent']
            if parent == -1:
                v_group.create_dataset("vecIn", data=[0.0, 0.0, 0.0])
            else:
                v_group.create_dataset("vecIn", data=graph[parent][node_id]['vecOut'])
            children = list(graph.successors(node_id))
            if len(children) > 0:
                for i, child_id in enumerate(children, 1):
                    edge_id = graph.edges[node_id, child_id]['edge_id']
                    v_group.create_dataset(f"child{i}", data=child_id)
                    v_group.create_dataset(f"edge{i}", data=edge_id)
                    v_group.create_dataset(f"vecOut{i}", data=graph[node_id][child_id]['vecIn'])

        # write edges
        edges = hf.create_group("edges")
        for u, v in graph.edges():
            edge_data = graph.edges[u, v]
            e_group = edges.create_group(f"e{edge_data['edge_id']}")
            e_group.create_dataset("pSource", data=u)
            e_group.create_dataset("pTarget", data=v)
            e_group.create_dataset("length", data=edge_data['length'])
            e_group.create_dataset("radius_avg", data=edge_data['radius_avg'])
            e_group.create_dataset("x", data=edge_data['x'])
            e_group.create_dataset("y", data=edge_data['y'])
            e_group.create_dataset("z", data=edge_data['z'])


def update_parent_nChild(graph):
    """update all nodes' parent and nChild attributes"""
    for node in graph.nodes():
        # update parent attribute
        predecessors = list(graph.predecessors(node))
        if not predecessors:  # No parent node（root）
            graph.nodes[node]['parent'] = -1
        else:  
            graph.nodes[node]['parent'] = predecessors[0]

        # update nChild attribute
        graph.nodes[node]['nChild'] = graph.out_degree(node)
    return graph


def merge_degree_one_nodes(graph):
    # create a copy
    g_new = graph.copy()

    # Collecting all the nodes need to be processed
    to_merge = [n for n in g_new.nodes() if g_new.out_degree(n) == 1]

    for node_1 in to_merge:
        # get parent node（only the first one, if there are many）
        predecessors = list(g_new.predecessors(node_1))
        if not predecessors:
            continue  
        node_0 = predecessors[0]

        # get the only child node
        successors = list(g_new.successors(node_1))
        if not successors:
            continue
        node_2 = successors[0]

        # get properties of the old edge
        edge_0_1 = g_new.get_edge_data(node_0, node_1)
        edge_1_2 = g_new.get_edge_data(node_1, node_2)
        if not edge_0_1 or not edge_1_2:
            continue

        if 'vecInS' in edge_1_2:
            new_attrs = {
                'edge_id': edge_0_1['edge_id'],
                'length': edge_0_1['length'] + edge_1_2['length'],
                'radius_avg': edge_0_1['radius_avg'],
                'vecIn': edge_0_1['vecIn'],
                'vecInS': edge_0_1['vecInS'],
                'vecOut': edge_1_2['vecOut'],
                'vecOutS': edge_1_2['vecOutS'],
                'x': np.concatenate([edge_0_1['x'], edge_1_2['x'][1:]]),  
                'y': np.concatenate([edge_0_1['y'], edge_1_2['y'][1:]]),
                'z': np.concatenate([edge_0_1['z'], edge_1_2['z'][1:]]),
            }
        else:
            new_attrs = {
                'edge_id': edge_0_1['edge_id'],
                'length': edge_0_1['length'] + edge_1_2['length'],
                'radius_avg': edge_0_1['radius_avg'],
                'vecIn': edge_0_1['vecIn'],
                'vecOut': edge_1_2['vecOut'],
                'x': np.concatenate([edge_0_1['x'], edge_1_2['x'][1:]]),  
                'y': np.concatenate([edge_0_1['y'], edge_1_2['y'][1:]]),
                'z': np.concatenate([edge_0_1['z'], edge_1_2['z'][1:]]),
            }

        # remove the old node and edge
        g_new.remove_edge(node_0, node_1)
        g_new.remove_edge(node_1, node_2)
        g_new.remove_node(node_1)

        # add new edge（keep the orignal edge ID）
        if not g_new.has_edge(node_0, node_2):
            g_new.add_edge(node_0, node_2, **new_attrs)
        else:
            existing = g_new[node_0][node_2]
            existing.update(new_attrs)
        # print(f"merge out_degree equals 1 node: {node_1}")

    return g_new


def split_degree_three_nodes(graph):
    g_new = graph.copy()
    node_list = [n for n in g_new.nodes() if g_new.out_degree(n) >= 3]
    if node_list:
        print('There are some nodes have out-degree more than 3.')
    for node in node_list:

        predecessors = list(g_new.predecessors(node))
        if not predecessors:
            continue 
        parent = predecessors[0]

        children = list(g_new.successors(node))
        edge_p_n = g_new.get_edge_data(parent, node)
        edge_n_c0 = g_new.get_edge_data(node, children[0])
        edge_n_c1 = g_new.get_edge_data(node, children[1])
        edge_n_c2 = g_new.get_edge_data(node, children[2])
        max_len = max(edge_n_c0['length'], edge_n_c1['length'], edge_n_c2['length'])

        for child in children:  # filter out unwanted spikes
            if g_new.out_degree(child) == 0:
                edge_n_c = g_new.get_edge_data(node, child)
                if edge_n_c['length']/max_len < 0.2 or edge_n_c['radius_avg']/edge_p_n['radius_avg'] < 0.2:
                    g_new.remove_edge(node, child)
                    g_new.remove_node(child)
                    print(f"removed edge{edge_n_c['edge_id']} ({node}, {child})")

        if g_new.out_degree(node) >= 3:
            min_od_child = children[0]
            min_out_degree = 10
            for child in children:
                if g_new.out_degree(node) <= min_out_degree:
                    min_od_child = child
                    min_out_degree = g_new.out_degree(node)

            e_id_dict = nx.get_edge_attributes(graph, 'edge_id')
            e_id_list = list(e_id_dict.values())
            max_id = max(e_id_list)

            if len(edge_p_n['x'] > 2):
                x0 = edge_p_n['x'][-2]
                y0 = edge_p_n['y'][-2]
                z0 = edge_p_n['z'][-2]
                new_edge_x = edge_p_n['x'][-2:]
                new_edge_y = edge_p_n['y'][-2:]
                new_edge_z = edge_p_n['z'][-2:]
            else:
                x0 = (edge_p_n['x'][-2] + edge_p_n['x'][-1])/2
                y0 = (edge_p_n['y'][-2] + edge_p_n['y'][-1])/2
                z0 = (edge_p_n['z'][-2] + edge_p_n['z'][-1])/2
                new_edge_x = np.array([x0, edge_p_n['x'][-1]])
                new_edge_y = np.array([y0, edge_p_n['y'][-1]])
                new_edge_z = np.array([z0, edge_p_n['z'][-1]])

            x1 = edge_p_n['x'][-1]
            y1 = edge_p_n['y'][-1]
            z1 = edge_p_n['z'][-1]
            dis = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)

            new_node_attr = {
                'parent': node,
                'nChild': 2,
                'x': x0,
                'y': y0,
                'z': z0,
            }
            if 'vecInS' in edge_p_n:
                new_edge_attr = {
                    'edge_id': max_id+1,
                    'length': dis,
                    'radius_avg': edge_p_n['radius_avg'],
                    'vecIn': edge_p_n['vecOut'],
                    'vecInS': edge_p_n['vecOutS'],
                    'vecOut': edge_p_n['vecOut'],
                    'vecOutS': edge_p_n['vecOutS'],
                    'x': new_edge_x,
                    'y': new_edge_y,
                    'z': new_edge_z,
                }
            else:
                new_edge_attr = {
                    'edge_id': max_id+1,
                    'length': dis,
                    'radius_avg': edge_p_n['radius_avg'],
                    'vecIn': edge_p_n['vecOut'],
                    'vecOut': edge_p_n['vecOut'],
                    'x': new_edge_x,
                    'y': new_edge_y,
                    'z': new_edge_z,
                }
            edge_p_n['length'] = abs(edge_p_n['length'] - dis)  # just in case the distance is negative when the length is small
            if len(edge_p_n['x'] > 2):
                edge_p_n['x'] = edge_p_n['x'][:-1]
                edge_p_n['y'] = edge_p_n['y'][:-1]
                edge_p_n['z'] = edge_p_n['z'][:-1]

            else:
                edge_p_n['x'] = np.array([edge_p_n['x'][0], x0])
                edge_p_n['y'] = np.array([edge_p_n['y'][0], y0])
                edge_p_n['z'] = np.array([edge_p_n['z'][0], z0])
            edge_n_moc = g_new.get_edge_data(node, min_od_child)
            edge_n_moc['x'] = np.insert(edge_n_moc['x'][2:], 0, x0)
            edge_n_moc['y'] = np.insert(edge_n_moc['y'][2:], 0, y0)
            edge_n_moc['z'] = np.insert(edge_n_moc['z'][2:], 0, z0)
            # remove old edges
            g_new.remove_edge(parent, node)
            g_new.remove_edge(node, min_od_child)

            new_node_id = max(g_new.nodes) + 1
            print(new_node_id)
            g_new.add_node(new_node_id, **new_node_attr)
            g_new.add_edge(parent, new_node_id, **edge_p_n)
            g_new.add_edge(new_node_id, node, **new_edge_attr)
            g_new.add_edge(new_node_id, min_od_child, **edge_n_moc)
            print(f"split out degree 3 node: {node}")

    update_parent_nChild(g_new)
    return g_new


def check_nodes(graph):
    root = [node for node, degree in graph.in_degree() if degree == 0][0]
    out_degrees = dict(graph.out_degree())
    degrees = list(out_degrees.values())
    target_nodes = [node for node, degree in out_degrees.items() if degree == 1]
    target_nodes = [x for x in target_nodes if x != root]
    if target_nodes:
        print("nodes with out-degree 1:", target_nodes)
    else:
        print("There is no out-degree 1 node except root.")
    target_nodes = [node for node, degree in out_degrees.items() if degree >= 3]
    if target_nodes:
        print("Nodes with out-degree 3:", target_nodes)
        for node in target_nodes:
            direct_children = list(graph.successors(node))
            for child in direct_children:
                edge_data = graph.get_edge_data(node, child)
                radius = edge_data['radius_avg']
                length = edge_data['length']
                print(f"边{edge_data['edge_id']} ({node}, {child}) radius:{radius: .3f}, length:{length: .3f}")
    else:
        print("There is no node has out-degree more than 3!")


def __remove_min_radius_leaf_edge(graph):
    # 1. Collect all the leaves
    leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    candidate_edges = []
    for leaf in leaf_nodes:
        predecessors = list(graph.predecessors(leaf))
        for pred in predecessors:
            edge_data = graph.get_edge_data(pred, leaf)
            if edge_data and 'radius_avg' in edge_data:
                candidate_edges.append((pred, leaf, edge_data['radius_avg']))

    if not candidate_edges:
        return graph 

    min_edge = min(candidate_edges, key=lambda x: x[2])
    pred_node, leaf_node, _ = min_edge

    graph.remove_edge(pred_node, leaf_node)
    graph.remove_node(leaf_node)

    return graph


def prune_tree(graph, node_num_keep):

    while len(graph.nodes) > node_num_keep:
        graph = __remove_min_radius_leaf_edge(graph)
        graph = merge_degree_one_nodes(graph)
    graph = update_parent_nChild(graph)
    return graph
