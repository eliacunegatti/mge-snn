import numpy as np
import igraph as ig
from numba import jit
from scipy.sparse import csc_matrix, diags, isspmatrix_csc, issparse


@jit(nopython=True)
def optimize_cnn(in_channels, out_channels, is_group, w_original, l1_split, l2_split,
                  conv_steps, kernel_size, stride, INPUT_SIZE):
    connection = []
    edge_weight = []
    connection.append((None, None))
    edge_weight.append(0.0)
    for j in (range(in_channels)):
        for c in range(out_channels):
            if is_group:
                w = w_original[j, 0]
                l = l1_split[j]
                r = l2_split[j]
            else:
                w = w_original[c, j]
                l = l1_split[j]
                r = l2_split[c]

            w = w.transpose().flatten()
            if np.count_nonzero(w) > 0:
                FINAL_KERNELS = []
                m = min(l)
                init = np.array([i + m for i in range(kernel_size)])
                start_init = init.copy()
                k_count, k_dim, t = 0, 0, 0
                while True:
                    k_count += 1
                    k_dim += 1
                    s = np.empty(0, dtype=np.int64)
                    for item in init:
                        appo = item
                        s = np.concatenate((s, np.arange(appo, appo + kernel_size * INPUT_SIZE, INPUT_SIZE)))
                        appo += INPUT_SIZE
                    FINAL_KERNELS.append(s)
                    if k_count == conv_steps:
                        break
                    if k_dim < int(np.sqrt(conv_steps)):
                        t += stride
                        init = start_init + t
                    else:
                        init = start_init + (INPUT_SIZE * stride)
                        start_init = init.copy()
                        k_dim, t = 0, 0

                for idx in range(len(FINAL_KERNELS)):
                    if idx >= len(r):
                        break
                    t1 = FINAL_KERNELS[idx]
                    ref = r[idx]
                    for id_w, item in enumerate(t1):
                        if w[id_w] != 0.0:
                            edge_weight.append(w[id_w])
                            connection.append((item, ref))
            if is_group:
                break

    return connection, edge_weight


@jit(nopython=True)
def igraph_edges_to_sparse_matrix(edges, num_vertices, mode='ALL'):
    if mode not in ('IN', 'OUT', 'ALL'):
        raise ValueError("Invalid mode. Use 'IN', 'OUT', or 'ALL'.")

    num_edges = len(edges)
    row_indices = np.empty(num_edges, dtype=np.int64)
    col_indices = np.empty(num_edges, dtype=np.int64)
    data = np.ones(num_edges, dtype=np.float64)

    if mode == 'IN':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = target
            col_indices[i] = source
    elif mode == 'OUT':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
    else: 
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
            


    return row_indices, col_indices, data


@jit(nopython=True)
def igraph_edges_to_sparse_matrix_weighted(edges, weights, mode='ALL'):
    if mode not in ('IN', 'OUT', 'ALL'):
        raise ValueError("Invalid mode. Use 'IN', 'OUT', or 'ALL'.")

    num_edges = len(edges)
    row_indices = np.empty(num_edges, dtype=np.int64)
    col_indices = np.empty(num_edges, dtype=np.int64)
    data = weights

    if mode == 'IN':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = target
            col_indices[i] = source
    elif mode == 'OUT':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
    else: 
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
        

    return row_indices, col_indices, data


def are_sparse_matrices_equal(matrix1, matrix2):
    # Check if both matrices are sparse CSC format
    if not isspmatrix_csc(matrix1) or not isspmatrix_csc(matrix2):
        print('not csc')
        return False

    # Check if the shapes are the same
    if matrix1.shape != matrix2.shape:
        print('noy shape')
        return False

    # Check if both matrices are non-empty 
    if not issparse(matrix1) or not issparse(matrix2):
        print('not sparse')
        return False


    if not np.array_equal(matrix1.data, matrix2.data):
        print('not data')
        print(matrix1.data[:10], matrix2.data[:10])
        print(np.where(matrix1.data != matrix2.data))
        return False
    if not np.array_equal(matrix1.indices, matrix2.indices):
        print('not indices')
        return False
    if not np.array_equal(matrix1.indptr, matrix2.indptr):
        print('not indptr')
        return False

    return True

def load_numba():
    G = ig.Graph.Erdos_Renyi(n=100, p=0.2, directed=True)
    G.es['weight'] = np.random.rand(G.ecount())
    G.es['weight'] = np.array(G.es['weight'], dtype=np.float32)
    num_vertices = int(G.vcount())
    edges = G.get_edgelist()
    edges = np.array(edges)
    row_indices, col_indices, data = igraph_edges_to_sparse_matrix(edges, num_vertices, mode='ALL')
    sparse_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))

    weights = np.array(G.es['weight'],dtype=np.float32)

    row_indices, col_indices, data = igraph_edges_to_sparse_matrix_weighted(edges, weights, mode='ALL')
    sparse_matrix_weighted = csc_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))

    sparse_adjacency = csc_matrix(G.get_adjacency_sparse().astype('f'))
    sparse_adjacency_weighted = csc_matrix(G.get_adjacency_sparse(attribute='weight').astype('f'))

    if are_sparse_matrices_equal(sparse_matrix, sparse_adjacency):
        pass
    else:
        Exception('The matrices are not equal')

    if are_sparse_matrices_equal(sparse_matrix_weighted, sparse_adjacency_weighted):
        pass
    else:
        Exception('The matrices are not equal')
    

    laplacian_matrix = csc_matrix(laplacian_from_adjacency(sparse_matrix))
    laplacian_matrix_weighted = csc_matrix(laplacian_from_adjacency(sparse_matrix_weighted))

    laplacian_matrix_igraph = csc_matrix(laplacian_from_adjacency(sparse_adjacency))
    laplacian_matrix_igraph_weighted = csc_matrix(laplacian_from_adjacency(sparse_adjacency_weighted))


    if are_sparse_matrices_equal(laplacian_matrix, laplacian_matrix_igraph):
        pass
    else:
        Exception('The matrices are not equal')
    differing_elements = np.where(laplacian_matrix.data != laplacian_matrix_igraph.data)
    if differing_elements[0].size == 0:
        pass
    else:
        Exception('The matrices are not equal')


    if are_sparse_matrices_equal(laplacian_matrix_igraph_weighted, laplacian_matrix_weighted):
        pass
    else:
        Exception('The matrices are not equal')
    differing_elements = np.where(laplacian_matrix_igraph_weighted.data != laplacian_matrix_weighted.data)
    if differing_elements[0].size == 0:
        pass
    else:
        Exception('The matrices are not equal')




def laplacian_from_adjacency(adjacency_matrix, degrees=None):
    adjacency_sparse = csc_matrix(adjacency_matrix)

    degrees = adjacency_sparse.sum(axis=0).A1.astype(float, casting='safe')  
    degree_matrix = diags(degrees, 0, format='csr')  

    laplacian_sparse = degree_matrix - adjacency_sparse

    return laplacian_sparse



