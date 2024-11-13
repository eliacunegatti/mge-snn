import math
from graph_encoding.utils_graph_analysis import *


def get_multipartite_graph(graph1, offset=1, plot=False):

    """
    """
    no_layers = len(graph1.layers)
    all_graphs = []
    residual_graphs_in, residual_graphs_out, index_residual_out, index_residual_in , pooling_downscaling = [], [], [], [], []
    correct_count = 0
    for idx_layer in range(0,no_layers-offset): 
        if (graph1.layers[idx_layer][1] == 'conv' and graph1.layers[idx_layer+1][1] == 'conv') or (graph1.layers[idx_layer][1] == 'linear' and graph1.layers[idx_layer+1][1] == 'linear'):
            G1 = graph1.get_graph(idx_layer, mode='igraph')
            G2 = graph1.get_graph(idx_layer+1, mode='igraph')
            l1, r1 = get_ids(G1)
            if idx_layer == 0:
                all_graphs.append(G1)
            l2, r2 = get_ids(G2)
            if (graph1.layers[idx_layer][1] == 'conv' and graph1.layers[idx_layer+1][1] == 'conv'):
                G2, _ = remove_padding(G2, graph1,idx_layer,r1,l2,r2)
            l2, r2 = get_ids(G2)
            all_graphs.append(G2)
            correct_count +=1

            if idx_layer+1+graph1.window_residual < no_layers:

                if graph1.layers[idx_layer+1+graph1.window_residual][1] == 'residual':
                    GR = graph1.get_graph(idx_layer+1+graph1.window_residual, mode='igraph')
                    residual_graphs_out.append(GR)
                    index_residual_out.append(correct_count+1)


        elif graph1.layers[idx_layer][1] == 'conv' and graph1.layers[idx_layer+1][1] == 'pooling':
            G1  = graph1.get_graph(idx_layer, mode='igraph')
            l1, r1 = get_ids(G1)
            pooled_results = pooling(r1, graph1, idx_layer)
            check = False
        

        elif (graph1.layers[idx_layer][1] == 'pooling' and graph1.layers[idx_layer+1][1] == 'conv') or  (graph1.layers[idx_layer][1] == 'pooling' and graph1.layers[idx_layer+1][1] == 'linear'):
            check_pooling = True
            G2  = graph1.get_graph(idx_layer+1, mode='igraph')
            g0, g1 = get_degree(G2)
            if (graph1.layers[idx_layer][1] == 'pooling' and graph1.layers[idx_layer+1][1] == 'conv'):
                G2, to_remove = remove_padding(G2, graph1,idx_layer,r1,l2,r2)
                assert len(g0)-len(to_remove) ==  len(pooled_results)

            all_graphs.append(G2)
            pooling_downscaling.append((len(all_graphs)-1,pooled_results))
            check = True


        elif (graph1.layers[idx_layer][1] == 'conv' and graph1.layers[idx_layer+1][1] == 'linear'):
            G1 = graph1.get_graph(idx_layer, mode='igraph')
            G2 = graph1.get_graph(idx_layer+1, mode='igraph')
            l1, r1 = get_ids(G1)
            l2, r2 = get_ids(G2)   
            if len(l2) == len(r1):
                try:
                    TG, G1_e, G1_v  = concatenate_bipartite_graphs(G1,G2)
                    check = True
                except:
                    check = False                        
            else:
                dim_ = int(math.sqrt(len(r1)/len(l2)))
                pooled_results = pooling(r1, graph1, idx_layer,(dim_,dim_), dim_)
                all_graphs.append(G2)
                pooling_downscaling.append((len(all_graphs)-1, pooled_results))

        elif (graph1.layers[idx_layer][1] == 'conv' and graph1.layers[idx_layer+1][1] == 'residual') or (graph1.layers[idx_layer][1] == 'linear' and graph1.layers[idx_layer+1][1] == 'residual'):
            G1 = graph1.get_graph(idx_layer, mode='igraph')
            GR = graph1.get_graph(idx_layer+1, mode='igraph')
            l1, r1 = get_ids(G1)
            lr, rr = get_ids(GR)

        elif (graph1.layers[idx_layer][1] == 'residual' and graph1.layers[idx_layer+1][1] == 'conv') or (graph1.layers[idx_layer][1] == 'residual' and graph1.layers[idx_layer+1][1] == 'linear'):
            correct_count += 1
            G2 = graph1.get_graph(idx_layer+1, mode='igraph')
            l2, r2 = get_ids(G2)
            if (graph1.layers[idx_layer][1] == 'residual' and graph1.layers[idx_layer+1][1] == 'conv'):
                G2, _ = remove_padding(G2, graph1,idx_layer,r1,l2,r2)
            all_graphs.append(G2)
            
            if idx_layer+1+graph1.window_residual < no_layers:

                if graph1.layers[idx_layer+1+graph1.window_residual][1] == 'residual':
                    GR = graph1.get_graph(idx_layer+1+graph1.window_residual, mode='igraph')
                    residual_graphs_out.append(GR)
                    index_residual_out.append(correct_count+1)


    dimension = []
    total_nodes = total_edges = 0
    dimension_2 = []
    count_pooling = 0
    for idx_layer in range(0,len(all_graphs)-1):
        if idx_layer == 0:
            TG = all_graphs[idx_layer]
            L1, R1 = get_ids(all_graphs[idx_layer])
            dimension.append(len(L1))
            dimension.append(len(R1))
            total_nodes += len(L1)+len(R1)
            total_edges += TG.ecount()
            dimension_2.append((len(L1), len(R1)))
    

        L1, R1 = get_ids(all_graphs[idx_layer+1])
        total_nodes += len(R1)
        total_edges += all_graphs[idx_layer+1].ecount()
        dimension.append(len(R1))
        dimension_2.append((len(L1), len(R1)))

        if dimension_2[idx_layer][1] != dimension_2[idx_layer+1][0] and pooling_downscaling[count_pooling][0] == idx_layer+1:
            pooling_elements = list(pooling_downscaling[count_pooling][1])
            min_value = min(pooling_elements[0])
            pooling_elements = [[x-min_value for x in item] for item in pooling_elements]  
            TG, _ , _ = concatenate_bipartite_graphs_all_pooling(TG,all_graphs[idx_layer+1], idx_layer+1, dimension, pooling_elements)
            count_pooling +=1
        else:
            TG, _ , _ = concatenate_bipartite_graphs_all(TG,all_graphs[idx_layer+1], idx_layer+1, dimension)

    TG = add_all_residual(TG,residual_graphs_out, index_residual_out, dimension,graph1.window_residual)
    
    if plot: plot_all_graphs(TG,dimension)  

    return TG, dimension