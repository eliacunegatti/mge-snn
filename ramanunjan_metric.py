
import numpy as np
import copy
import collections
import math
import scipy as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from graph_encoding.numba_utils import igraph_edges_to_sparse_matrix, igraph_edges_to_sparse_matrix_weighted


#For the ramanunjan metric we extend the original codebase https://github.com/VITA-Group/ramanujan-on-pai

### Ramanujan metric ####
def iterative_mean_spectral_gap(G, l1,r1):
    """Calculate the total variance of bounds on every sub-graph of layers that has at-least
    d-in/out degree


    """

    direct_both_results = []
    d_left_graphs = find_d_left_regular(G,l1,r1)
    for left_regular_subgraph in d_left_graphs:
        ram_scores = ramanujan_score(left_regular_subgraph)
        if ram_scores[0][0] == None:  # == (-1, -1, -1, 0, -1):
            continue
        else:
            direct_both_results.append(ram_scores)


    if len(direct_both_results) == 0:
        return [[None, None]]
    else:
        directed_both_return = np.mean(direct_both_results, axis=0)


    return directed_both_return
    

def filter_zero_degree(graph, dim_in):
    """
    Filter zero-degree nodes to not affect d_avg_l/d_avg_r.
    """
    # Calculate the degrees of nodes in the igraph graph
    degrees = graph.degree()
    # Identify non-zero degree nodes
    non_zero_nodes = [i for i, degree in enumerate(degrees) if degree > 0]

    # Check if all nodes have non-zero degrees
    if len(non_zero_nodes) == graph.vcount():
        return graph, dim_in

    # Create a subgraph by filtering out zero-degree nodes
    subgraph = graph.subgraph(non_zero_nodes)

    return subgraph, dim_in



def find_d_left_regular(G, l1, r1,  mindegree: int = 3):
    """Return a subgraph of a bipartite graph with a minimum in/out degree.

    :param layer: A dictionary containing the graph and other information.
    :param mindegree: The minimum in/out degree required.
    :returns: A list of subgraphs that meet the degree criteria.
    """
    graph = copy.deepcopy(G)
    num_nodes = graph.vcount()
    degrees = graph.degree()
    
    d_l = degrees[:l1]
    subgraphs = []
    degrees = collections.Counter(d_l)
    for degree, count in degrees.items():
        if degree < mindegree: continue
        node_indices = [i for i, d in enumerate(d_l) if d == degree]
        right_nodes = list(range(l1, num_nodes))
        nodes_to_keep = node_indices + right_nodes
        subgraph = graph.induced_subgraph(nodes_to_keep, implementation="create_from_scratch")
        #subgraph.to_undirected()
        #subgraph.to_directed()
        subgraphs.append({
            'dim_in': count,
            'graph': subgraph, 
            'dim_out': num_nodes - count,
        })

    return subgraphs


def get_eig_values(matrix: np.array, k: int = 3):
    """
    get the real eig of a square matrix
    for bi-graph, the third largest eig denotes connectivity
    """

    adj_eigh_val, _ = sp.sparse.linalg.eigsh(matrix, k=3, which='LM')
    abs_eig = [abs(i) for i in adj_eigh_val]
    abs_eig.sort(reverse=True)
    return abs_eig



def delta_r(avg_deg_left: float, avg_deg_right: float,
            eig_second: float) -> float:
    """
    calc the change in degree
    """
    ub = math.sqrt(avg_deg_left - 1) + math.sqrt(avg_deg_right - 1)
    return ub - eig_second, ub






def get_all_metrics(graph, layer, degree, dim_in, d_avg_l, d_avg_r):
    weights = np.array(graph.es['weight'], dtype=np.float32)
    edges = np.array(graph.get_edgelist())
    num_vertices = graph.vcount()

    # Create weighted adjacency matrix
    row_indices, col_indices, data = igraph_edges_to_sparse_matrix_weighted(edges, weights, mode='ALL')
    adj_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))

    # Calculate spectral gap
    m_eig_vals = get_eig_values(adj_matrix)
    t2_m, t1_m = m_eig_vals[-1], m_eig_vals[0]
    spectral_gap = t1_m - t2_m
    del adj_matrix

    row_indices, col_indices, data = igraph_edges_to_sparse_matrix(edges, num_vertices, mode='ALL')
    adj_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))
    m_eig_vals = get_eig_values(adj_matrix)
    t2_m, t1_m = m_eig_vals[-1], m_eig_vals[0]

    rm_sm, rm_ub_sm = delta_r(d_avg_l, d_avg_r, t2_m)


    del adj_matrix
    return spectral_gap, rm_sm / rm_ub_sm

        
def ramanujan_score(layer):

    graph = layer['graph']
    dim_in = layer['dim_in']
    graph, dim_in = filter_zero_degree(graph, dim_in)
    degree = graph.degree()

    d_avg_l = np.mean(degree[0:dim_in])
    d_avg_r = np.mean(degree[dim_in:])
    
    if d_avg_l >= 3 and d_avg_r >= 3:    
        graph = graph.as_undirected(combine_edges='first')
        graph.to_undirected()
        graph.to_directed()
        sg_direct_both, delta_r_direct_both = get_all_metrics(graph, layer, degree, dim_in, d_avg_l, d_avg_r)
    else:
        return [[None,None]] 

    return [[sg_direct_both, delta_r_direct_both]]


def filter_zero_degree_general(graph, dim_in):
    """
    Filter zero-degree nodes to not affect d_avg_l/d_avg_r.
    """
    # Calculate the degrees of nodes in the igraph graph
    degrees = graph.degree()
    # Identify non-zero degree nodes
    non_zero_nodes = [i for i, degree in enumerate(degrees) if degree > 0]
    dim_in = dim_in - degrees[0:dim_in].count(0)
    dim_out = len(non_zero_nodes) - dim_in
    # Check if all nodes have non-zero degrees
    if len(non_zero_nodes) == graph.vcount():
        return graph, dim_in, dim_out

    # Create a subgraph by filtering out zero-degree nodes
    subgraph = graph.subgraph(non_zero_nodes)
    return subgraph, dim_in, dim_out