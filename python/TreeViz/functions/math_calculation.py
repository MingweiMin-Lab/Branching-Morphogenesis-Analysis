import networkx as nx
import numpy as np
from scipy.interpolate import splprep, splev
import sys
sys.path.append("./functions") 
import powersmooth2 as ps


def get_included_angle(vec1, vec2):
    angle = np.acos(vec1.dot(vec2))
    angle = angle * 180 / np.pi
    return angle


def get_vector_plane_angle(v1, v2, v3):
    # calculate normal vector
    normal = np.cross(v1, v2)
    if np.allclose(normal, [0, 0, 0]):
        raise ValueError("v1 and v2 are collinear and therefore cannot define/form a plane.")

    # Calculate the angle between normal and v3
    dot_product = np.abs(np.dot(normal, v3))
    norm_normal = np.linalg.norm(normal)
    norm_v3 = np.linalg.norm(v3)

    cos_phi = dot_product / (norm_normal * norm_v3)
    phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))

    # radian to degree
    theta = np.degrees(np.pi / 2 - phi)

    return theta


def compute_max_descendant_distance(graph):
    if not nx.is_directed_acyclic_graph(graph):
        # deal with the possible loop
        scc = list(nx.strongly_connected_components(graph))
        condensed_G = nx.condensation(graph, scc)
        raise NotImplementedError("Remove loop")

    topo_order = list(nx.topological_sort(graph))
    max_distances = {}

    for node in graph.nodes():
        dist = {n: -float('inf') for n in graph.nodes()}
        dist[node] = 0
        for u in topo_order:
            for v in graph.successors(u):
                edge_length = graph.edges[u, v].get('length', 0)
                if dist[v] < dist[u] + edge_length:
                    dist[v] = dist[u] + edge_length
        max_dist = max(dist.values())
        max_distances[node] = max_dist if max_dist != 0 else 0  # return 0 if there is no Child

    nx.set_node_attributes(graph, max_distances, 'distance_to_leaves')

    # for idx, node in enumerate(graph.nodes()):
    #     graph[node]['distance_to_leaves'] = max_distances[node]

    return max_distances

    
def calc_node_type_l1(graph, smooth_edge: bool = True):
    
    root = [node for node, degree in graph.in_degree() if degree == 0][0]

    if smooth_edge:
        graph = recalculate_direction_vec(graph)
    len_dict = nx.get_edge_attributes(graph, 'length')
    len_arr = np.array(list(len_dict.values()))
    median_len = np.median(len_arr)
    # print(median_len)
    graph.nodes[root]['node_type_l1'] = -1  # -1 means not considering
    children0 = list(graph.successors(root))
    for child in children0:
        graph[root][child]['edge_type_l1'] = -1  # -1 means not considering
    nodes = [edge[1] for edge in graph.edges]

    max_distances = compute_max_descendant_distance(graph)
    for node in nodes:
        node2test = -1

        node_type = -1
        children = list(graph.successors(node))
        if len(children) == 2:
            node_type = 1  # type 1 is 'bifurcation/clefting'
            parent = list(graph.predecessors(node))[0]

            len0 = graph[node][children[0]]['length']
            len1 = graph[node][children[1]]['length']

            node_to_leaves = max_distances[node]
            len0_to_leaves = max_distances[children[0]]
            len1_to_leaves = max_distances[children[1]]
            if node == node2test:
                print(f'len0 is {len0: .2f}, len1 is {len1: .2f}')
                print(f'len0_to_leaves is {len0_to_leaves: .2f}, len1_to_leaves is {len1_to_leaves: .2f}')

            if graph.out_degree(children[0]) * graph.out_degree(children[1]) > 0 and len1_to_leaves/len0_to_leaves > 4:
                node_type = 0  # this is a branching/budding node
                graph[node][children[0]]['edge_type_l1'] = 2  # means branched stem
                graph[node][children[1]]['edge_type_l1'] = 0  # means main stem
            elif graph.out_degree(children[0]) * graph.out_degree(children[1]) > 0 and len0_to_leaves/len1_to_leaves > 4:
                node_type = 0  # this is a branching/budding node
                graph[node][children[0]]['edge_type_l1'] = 0  # means main stem
                graph[node][children[1]]['edge_type_l1'] = 2  # means branched stem
            elif graph.out_degree(children[0]) == 0 and graph.out_degree(children[1]) > 0 and len0/len1_to_leaves < 0.5:
                node_type = 0  # this is a branching/budding node
                graph[node][children[0]]['edge_type_l1'] = 2  # means branched stem
                graph[node][children[1]]['edge_type_l1'] = 0  # means main stem
            elif graph.out_degree(children[1]) == 0 and graph.out_degree(children[0]) > 0 and len1/len0_to_leaves < 0.5:
                node_type = 0  # this is a branching/budding node
                graph[node][children[0]]['edge_type_l1'] = 0  # means main stem
                graph[node][children[1]]['edge_type_l1'] = 2  # means branched stem
            else:
                r0 = graph[node][children[0]]['radius_avg']
                r1 = graph[node][children[1]]['radius_avg']

                if smooth_edge:
                    vecIn = graph[parent][node]['vecOutS']
                    vecOut1 = graph[node][children[0]]['vecInS']
                    vecOut2 = graph[node][children[1]]['vecInS']
                else:
                    vecIn = graph[parent][node]['vecOut']
                    vecOut1 = graph[node][children[0]]['vecIn']
                    vecOut2 = graph[node][children[1]]['vecIn']
                ang12 = abs(get_included_angle(vecOut1, vecOut2))
                if node == node2test:
                    print(f'ang12 is {ang12: .2f}')
                arr = [r0, r1]
                min_val = min(arr)
                min_index = arr.index(min_val)
                max_val = max(arr)
                max_index = arr.index(max_val)
                if min_val > 0:
                    r_ratio_diff = (max_val - min_val) / min_val
                else:
                    r_ratio_diff = 0
                if node == node2test:
                    print(f'r_ratio_diff is {r_ratio_diff}')
                if r_ratio_diff > 0.5:
                    node_type = 0  # this is a branching/budding node
                    graph[node][children[min_index]]['edge_type_l1'] = 2  # means branched stem
                    graph[node][children[max_index]]['edge_type_l1'] = 0  # means main stem
                # elif ang12 > 120:
                #     node_type = 1  # this is a clefting node
                #     graph[node][children[0]]['edge_type_l1'] = 1
                #     graph[node][children[1]]['edge_type_l1'] = 1
                elif len0 / len1 < 0.5:
                    node_type = 0  # this is a branching/budding node
                    graph[node][children[0]]['edge_type_l1'] = 2  # means branched stem
                    graph[node][children[1]]['edge_type_l1'] = 0  # means main stem
                elif len0 / len1 > 2:
                    node_type = 0  # this is a branching/budding node
                    graph[node][children[0]]['edge_type_l1'] = 0  # means main stem
                    graph[node][children[1]]['edge_type_l1'] = 2  # means branched stem
                else:
                    ang0 = abs(get_included_angle(vecIn, vecOut1))
                    ang1 = abs(get_included_angle(vecIn, vecOut2))
                    if node == node2test:
                        print(f'ang0 is {ang0: .2f}, ang1 is {ang1: .2f}, ang12 is {ang12: .2f}')
                        print(f'vIn:{vecIn}, vO1:{vecOut1}, v02:{vecOut2}')

                    if abs(ang0 - ang1) > 45 or (ang0 < 20 and (ang1 - ang0 > 20)) or (ang1 < 20 and (ang0 - ang1 > 20)):
                        arr = [ang0, ang1]
                        min_val = min(arr)
                        min_index = arr.index(min_val)
                        max_val = max(arr)
                        max_index = arr.index(max_val)
                        node_type = 0  # this is a branching/budding node
                        graph[node][children[min_index]]['edge_type_l1'] = 2  # means branched stem
                        graph[node][children[max_index]]['edge_type_l1'] = 0  # means main stem
                    else:
                        node_type = 1  # type 1 is 'bifurcation/clefting'
                        graph[node][children[min_index]]['edge_type_l1'] = 1
                        graph[node][children[max_index]]['edge_type_l1'] = 1
        graph.nodes[node]['node_type_l1'] = node_type
    return graph


def calc_edge_type_l2(graph, smooth_edge: bool = True):
    root = [node for node, degree in graph.in_degree() if degree == 0][0]
    angle_limit = 30
    graph = calc_node_type_l1(graph, smooth_edge)
    vecName = 'vecInS'
    if not smooth_edge:
        vecName = 'vecIn'
    children0 = list(graph.successors(root))
    for child in children0:
        graph[root][child]['edge_type_l2'] = -1  # -1 means meaningless
        graph[root][child]['dihedral_angle'] = -1 # -1 means meaningless
    l1_dict = nx.get_node_attributes(graph, 'node_type_l1')
    nodes = [edge[1] for edge in graph.edges]
    for node in nodes:
        edge_type_l2 = -1
        children = list(graph.successors(node))
        if len(children) == 2:
            if l1_dict[node] == 0:  # first node is Budding
                for idx, child in enumerate(children):
                    if l1_dict[child] == 0:  # next node is Budding
                        if graph[node][child]['edge_type_l1'] == 0: # main stem
                            edge_type_l2 = 0
                        else: # branch
                            edge_type_l2 = 2
                        vec0 = graph[node][child][vecName]
                        other_child = children[1 - idx]
                        vec1 = graph[node][other_child][vecName]
                        grandchildren = list(graph.successors(child))
                        if graph[child][grandchildren[0]]['edge_type_l1'] == 0:
                            vec2 = graph[child][grandchildren[1]][vecName]
                        else:
                            vec2 = graph[child][grandchildren[0]][vecName]
                        angle = get_vector_plane_angle(vec0, vec1, vec2)
                        if angle > angle_limit:
                            edge_type_l2 = edge_type_l2+1

                    elif l1_dict[child] == 1: # next node is Clefting
                        if graph[node][child]['edge_type_l1'] == 0:  # main stem
                            edge_type_l2 = 4
                        elif graph[node][child]['edge_type_l1'] == 2:  # branch
                            edge_type_l2 = 6
                        vec0 = graph[node][child][vecName]
                        other_child = children[1 - idx]
                        vec1 = graph[node][other_child][vecName]
                        grandchildren = list(graph.successors(child))
                        vec2 = graph[child][grandchildren[0]][vecName]
                        angle = get_vector_plane_angle(vec0, vec1, vec2)
                        if angle > angle_limit:
                            edge_type_l2 = edge_type_l2 + 1
                    else:
                        edge_type_l2 = -1  # -1 means meaningless
                        angle = -1  #-1 means dihedral angle is meaningless
                    graph[node][child]['edge_type_l2'] = edge_type_l2
                    graph[node][child]['dihedral_angle'] = angle
            elif l1_dict[node] == 1:  # first node is Clefting
                for idx, child in enumerate(children):
                    if l1_dict[child] == 0:  # next node is Budding
                        edge_type_l2 = 8
                        vec0 = graph[node][child][vecName]
                        other_child = children[1 - idx]
                        vec1 = graph[node][other_child][vecName]
                        grandchildren = list(graph.successors(child))
                        if graph[child][grandchildren[0]]['edge_type_l1'] == 0:
                            vec2 = graph[child][grandchildren[1]][vecName]
                        else:
                            vec2 = graph[child][grandchildren[0]][vecName]
                        angle = get_vector_plane_angle(vec0, vec1, vec2)
                        if angle > angle_limit:
                            edge_type_l2 = edge_type_l2+1
                    elif l1_dict[child] == 1:  # next node is clefting
                        edge_type_l2 = 10
                        vec0 = graph[node][child][vecName]
                        other_child = children[1 - idx]
                        vec1 = graph[node][other_child][vecName]
                        grandchildren = list(graph.successors(child))
                        vec2 = graph[child][grandchildren[0]][vecName]
                        angle = get_vector_plane_angle(vec0, vec1, vec2)
                        if angle > angle_limit:
                            edge_type_l2 = edge_type_l2+1
                    else:
                        edge_type_l2 = -1
                        angle = -1  #-1 means dihedral angle is meaningless
                    graph[node][child]['edge_type_l2'] = edge_type_l2
                    graph[node][child]['dihedral_angle'] = angle
            else:
                edge_type_l2 = -1
                angle = -1  # -1 means dihedral angle is meaningless
                for idx, child in enumerate(children):
                    graph[node][child]['edge_type_l2'] = edge_type_l2
                    graph[node][child]['dihedral_angle'] = angle
    return graph


def get_tree_kernel_composition_vec(G, normalized = True):

    l1_dict = nx.get_node_attributes(G, 'node_type_l1')
    l1_list = list(l1_dict.values())
    l2_dict = nx.get_edge_attributes(G, 'edge_type_l2')
    l2_list = list(l2_dict.values())
    tree_vec = []
    for i in range(0, 2):
        tree_vec.append(l1_list.count(i))

    for i in range(0, 12):
        tree_vec.append(l2_list.count(i))
    if normalized:
        tree_vec = np.array(tree_vec, dtype=np.float64)
        tree_vec[0:2] = 0.5 * tree_vec[0:2] / sum(tree_vec[0:2])
        tree_vec[2:] = 0.5 * tree_vec[2:] / sum(tree_vec[2:])
    return tree_vec


def get_endpoint_tangents(x, y, z, normalize=True):

    # combine xyz into a (3, n) array
    # points = np.array([x, y, z])
    points = np.vstack([x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)])

    # B spline interpolation（s=0 corresponds to exact interpolation）
    tck, u, *_ = splprep(points, s=0)

    # derivative at start and ending points
    der_start = np.array(splev(u[0], tck, der=1))
    der_end = np.array(splev(u[-1], tck, der=1))

    if normalize:
        norm_start = np.linalg.norm(der_start)
        norm_end = np.linalg.norm(der_end)
        if norm_start > 0:
            der_start /= norm_start
        if norm_end > 0:
            der_end /= norm_end

    return der_start, der_end


def recalculate_direction_vec(graph):
    # smooth the edges first and then calculate edge direction
    for u, v, data in graph.edges(data=True):
        vecIn = data['vecIn']
        vecOut = data['vecOut']
        if len(data['x']) < 5:
            graph.edges[u, v]['vecInS'] = vecIn
            graph.edges[u, v]['vecOutS'] = vecOut
        else:
            x_smooth = ps.powersmooth2(data['x'], 2, 100)
            y_smooth = ps.powersmooth2(data['y'], 2, 100)
            z_smooth = ps.powersmooth2(data['z'], 2, 100)
            vecInS, vecOutS = get_endpoint_tangents(x_smooth, y_smooth, z_smooth)
            graph.edges[u, v]['vecInS'] = vecInS
            graph.edges[u, v]['vecOutS'] = vecOutS
    return graph

# demo
if __name__ == "__main__":
    # create a spiral curve
    t = np.linspace(0, 2 * np.pi, 50)
    x = np.cos(t)
    y = np.sin(t)
    z = t

    start_tan, end_tan = get_endpoint_tangents(x, y, z)

    print("Tangent at start point:", start_tan)
    print("Tangent at ending point:", end_tan)
