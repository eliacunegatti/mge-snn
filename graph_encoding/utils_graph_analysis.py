#import PyTorch libraries
import torch 
import torch.nn as nn
import torchvision.transforms as transforms

# import graph libraries
import igraph as ig
import networkx as nx


import numpy as np
import matplotlib.pyplot as plt


def get_degree(G):
        a_out, a_in = [], [] 
        for item in G.vs:
            if item['type'] == False:
                a_out.append(item.degree(mode='out'))
            elif item["type"] == True:
                a_in.append(item.degree(mode='in'))
        return np.array(a_out), np.array(a_in)

def get_ids(G):
        a_out, a_in = [], [] 
        for item in G.vs:
            if item['type'] == False:
                a_out.append(item.index)
            elif item["type"] == True:
                a_in.append(item.index)
        return np.array(a_out), np.array(a_in)



def get_dim(g, index):
        dim =  g.fmap_sizes[index][0] - (2*g.layers[index][0].padding[0])
        I = torch.rand(dim, dim)
        transform = transforms.Pad(g.layers[index][0].padding[0])
        I = transform(I)        
        I = torch.flatten(I)
        return I


def connectivity_1(graph1, g2, g1, I, index):
    to_remove = []
    check = []
    idk = 0
    for i in range(graph1.layers[index][0].in_channels):
        k = g2[idk:len(I)+idk]

        idk += len(I)
        for jf in range(len(I)):
            if I[jf] == 0:
                to_remove.append(k[jf])
            else:
                check.append(k[jf])


    if len(g1) != len(check):
        raise Exception('something went wrong')
    return to_remove
        


def connectivity_2(graph1, g2,g1, I, index):
        to_remove = []
        check = []
        idk = 0
        for i in range(graph1.layers[index][0].in_channels):
            k = g2[idk:len(I)+idk]
            idk += len(I)
            for jf in range(len(I)):
                if I[jf] == 0:
                    to_remove.append(k[jf])
                else:
                    check.append(k[jf])
        return to_remove



def pooling(g1, graph1, idx, window_size=None, stride=None):
    t = torch.from_numpy(g1)
    t = t.to(torch.float32)
    unflatten = nn.Unflatten(-1,(graph1.layers[idx][0].out_channels, graph1.fmap_sizes[idx][1], graph1.fmap_sizes[idx][1]))
    input_tensor = unflatten(t)

    if window_size == None: window_size = graph1.layers[idx+1][0].kernel_size 
    if stride == None: stride = graph1.layers[idx+1][0].stride[0]

    pooled_results = [[] for i in range(input_tensor.size(0))]

    # Iterate over the tensor and perform pooling
    for i in range(0, input_tensor.size(1), int(stride)):
        for j in range(0, input_tensor.size(2), int(stride)):
            window = input_tensor[:, i:i+window_size[0], j:j+window_size[1]]
            for w in range(window.size(0)):
                pooled_results[w].append(torch.flatten(window[w]).tolist())


    #flatten the list
    pooled_results = [item for sublist in pooled_results for item in sublist]

    return pooled_results



def concatenate_bipartite_graphs(G1,G2):
    l1, r1 = get_ids(G1)
    l2, r2 = get_ids(G2)
    
    g1_e = G1.ecount()
    g1_v = G1.vcount()
    g1_w =  G1.es['weight']
    count = [i_ + len(l1) + len(r1) for i_, item in enumerate(G2.vs) if item['type'] == True]
    G1.add_vertices(count)

    #get attribute weight of G1

    # Create edges to add from G2 to TG
    edges_to_add = [(edge.source + len(l1), edge.target + len(l1)) for edge in G2.es]
    
    # Add the edges to TG in one step
    G1.add_edges(edges_to_add)
    G1.es['weight'] = g1_w + G2.es['weight']
    if G1.vcount() != len(l1)+len(r1)+len(r2) or G1.ecount() != g1_e+G2.ecount():
        raise Exception('Something went wrong')
    else:
        pass
    return G1, g1_e, g1_v

def concatenate_bipartite_graphs_all_pooling(G1,G2, idx_layer, dim, pooled_):
    l1, r1 = get_ids(G1)
    l2, r2 = get_ids(G2)
    
    g1_e = G1.ecount()
    g1_v = G1.vcount()
    g1_w =  G1.es['weight']
    offset = sum(dim[:idx_layer])
    pooled_ = [[int(x+offset) for x in item] for item in pooled_]
    offset_2 = max(pooled_[-1])+1

    count = [i_ + offset_2 - len(l2)  for i_, item in enumerate(G2.vs) if item['type'] == True]
    G1.add_vertices(count)
    g1_w = G1.es['weight']
    edges_to_add, weights = [], []
    for item in G2.es():
        nodes_left = pooled_[item.source]

        
        node_right = count[item.target-len(l2)]
        w = item['weight']
        for node in nodes_left:
            edges_to_add.append((node, node_right))
            weights.append(w)
    
    G1.add_edges(edges_to_add)
    G1.es['weight'] = g1_w + weights
    
    return G1, g1_e, g1_v


def concatenate_bipartite_graphs_all(G1,G2, idx_layer, dim):
    
    g1_e = G1.ecount()
    g1_v = G1.vcount()
    g1_w =  G1.es['weight']
    offset = sum(dim[:idx_layer])

    count_mask = np.array(G2.vs['type']) == True
    count = np.arange(len(G2.vs))[count_mask] + offset

    # Add new vertices to G1
    G1.add_vertices(len(count))

    #get attribute weight of G1
    # Create edges to add from G2 to TG
    edges_to_add = [(edge.source + offset, edge.target + offset) for edge in G2.es]

    try:
        G1.add_edges(edges_to_add)
        G1.es['weight'] = g1_w + G2.es['weight']
    
        if G1.ecount() != g1_e+G2.ecount():
            raise Exception('Something went wrong')
        else:
            pass
    except:
        edges_to_add = [(edge.source + offset, edge.target + offset) for edge in G2.es if edge.source + offset < offset+len(count) and edge.target + offset < offset+len(count)]
        G1.add_edges(edges_to_add)
        G1.es['weight'] = g1_w + G2.es['weight']

    return G1, g1_e, g1_v


def concatenate_residual_incoming(TG, GR, l1, r1, lr, rr, l2, r2):
    #GET MAX ID OF TG
    max_vertex_id = TG.vcount()
    count = 0
    weights = TG.es['weight']

    TG.add_vertices([max_vertex_id + item.index for item in GR.vs if not item["type"]])

    edges_to_add = [(edge.source + max_vertex_id, len(l1) + (edge.target - len(lr))) for edge in GR.es]
    TG.add_edges(edges_to_add)
    TG.es['weight'] = weights + GR.es['weight']

    return TG

def concatenate_residual_out(TG, GR, l1, r1, lr, rr, l2, r2):
    #GET MAX ID OF TG
    max_vertex_id = TG.vcount()
    count = 0
    weights = TG.es['weight']
    TG.add_vertices([max_vertex_id + item.index for item in GR.vs if item["type"]])

    edges_to_add = [(edge.source + len(l1), edge.target+len(l1)+len(r2)) for edge in GR.es]

    TG.add_edges(edges_to_add)
    TG.es['weight'] = weights + GR.es['weight']

    return TG
def remove_padding(G2, graph1,idx_layer,r1,l2,r2):
    I = get_dim(graph1, idx_layer+1)

    if int((I.numel() - torch.count_nonzero(I))*graph1.layers[idx_layer+1][0].in_channels) == len(l2)- len(r1):
        to_remove = connectivity_1(graph1,l2, r1, I, idx_layer+1)
          
    else:                               
        to_remove = connectivity_2(graph1,l2, r2, I,idx_layer+1)
          
    G2.delete_vertices(to_remove)
    return G2, to_remove



def concatenate_bipartite_graphs_with_pooling(G1,G2, pooled_results):

    TG = ig.Graph(directed=True)
    layer1, layer2, layer3 = [], [], []

    # Define a function to add vertices and update layers
    def add_vertex_and_update_layer(graph, item, layer, offset):
        vertex_index = item.index + offset
        graph.add_vertex(vertex_index)
        layer.append(vertex_index)
        return vertex_index

    layer1 = [add_vertex_and_update_layer(TG, item, layer1, offset=0) for item in G1.vs if not item['type']]
    layer2 = [add_vertex_and_update_layer(TG, item, layer2, offset=max(layer1)+1) for item in G2.vs if not item['type']]
    layer3 = [add_vertex_and_update_layer(TG, item, layer3, offset=max(layer2)+1) for item in G2.vs if item['type']]


    list_edges = [(int(e.source), int(node)) for count, node in enumerate(range(min(layer2), max(layer2) + 1))
                    for n in pooled_results[count]
                    for e in [G1.es[edge_idx] for edge_idx in G1.incident(int(n), mode="in")]]

    TG.add_edges(list_edges)
    
    max_layer1 = max(layer1)
    edges_to_add = [(edge.source + max_layer1 + 1, edge.target + max_layer1 + 1) for edge in G2.es]
    TG.add_edges(edges_to_add)
    TG.es["weight"] = G1.es["weight"] + G2.es["weight"]
    return TG, layer1, layer2, layer3

def plot_rolled_graph(G,l1,r1):
    #from igraph to networkx
    print('Generating graph')
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(G.vs)))

    # Iterate through the edges in the igraph graph
    for edge in G.es:
        source = edge.source
        target = edge.target
        graph.add_edge(source, target)
    for node in graph.nodes(data=True):
        #get weighted degree of nodes
        node[1]["degree"] = graph.degree(node[0], weight='weight')
        if node[0] < l1:
            node[1]["layer"] = 0
        elif node[0] >= l1:
            node[1]["layer"] = 1
 
    print('pos -...')

    pos = nx.multipartite_layout(graph, subset_key='layer')
    plt.figure(figsize=(10, 5))
    print('plotting -...')

    
    nx.draw(graph, pos, node_color='red', with_labels=False,node_size=1,edge_color='grey',arrows=False, width=1)
    plt.show()
    plt.axis('off')
    plt.show()

def plot_residual_in(G,l1,r1,r2, lr):
    print(G.vcount(), G.ecount())
    print('Generating graph')
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(G.vs)))

    # Iterate through the edges in the igraph graph
    for edge in G.es:
        source = edge.source
        target = edge.target
        weight = edge["weight"]
        graph.add_edge(source, target, weight=weight)
    for node in graph.nodes(data=True):
        node[1]["degree"] = graph.degree(node[0], weight='weight')
        if node[0] < len(l1):
            node[1]["layer"] = 1
        elif node[0] >= len(l1) and node[0] < len(l1)+len(r1):
            node[1]["layer"] = 2
        elif node[0] >= len(l1)+len(r1) and node[0] < len(l1)+len(r1)+len(r2):
            node[1]["layer"] = 3
        else:
            node[1]["layer"] = 0

    print('pos -...')

    pos = nx.multipartite_layout(graph, subset_key='layer')
    plt.figure(figsize=(10, 5))
    print('plotting -...')

    nx.draw(graph, pos, node_color='red', with_labels=False,node_size=1,edge_color='grey',arrows=False, width=0.05)
    plt.savefig(f'residual-{len(l1)}_in.png')


def plot_residual_out(G,l1,r1,r2, lr):
    print(G.vcount(), G.ecount())
    print('Generating graph')
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(G.vs)))

    # Iterate through the edges in the igraph graph
    for edge in G.es:
        source = edge.source
        target = edge.target
        weight = edge["weight"]
        graph.add_edge(source, target, weight=weight)
    count0 = count1= count2= count3= 0
    for node in graph.nodes(data=True):
        node[1]["degree"] = graph.degree(node[0], weight='weight')
        if node[0] < len(l1):
            node[1]["layer"] = 0
            count0 += 1
        elif node[0] >= len(l1) and node[0] < len(l1)+len(r1):
            node[1]["layer"] = 1
            count1 += 1
        elif node[0] >= len(l1)+len(r1) and node[0] < len(l1)+len(r1)+len(r2):
            node[1]["layer"] = 2
            count2 += 1
        else:
            count3 +=1
            node[1]["layer"] = 3
    print(count0, count1, count2, count3)


    print(graph.number_of_nodes(), graph.number_of_edges())
    print('pos -...')

    pos = nx.multipartite_layout(graph, subset_key='layer')
    plt.figure(figsize=(10, 5))
    print('plotting -...')

    nx.draw(graph, pos, node_color='red', with_labels=False,node_size=1,edge_color='grey',arrows=False, width=0.05)
    plt.savefig(f'residual-{len(l1)}_out.png')


def plot_all_graphs(G,dimension):
    print(G.vcount(), G.ecount())

    print('Generating graph')
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(G.vs)))

    connection = []
    for edge in G.es:
        source = edge.source
        target = edge.target
        #connection.append((source, target))
        if edge["weight"] == 'residual':
            connection.append((source, target, 'dashed'))
        else:
            connection.append((source, target, 'solid'))


    graph.add_weighted_edges_from(connection)
    nodes = graph.nodes(data=False)
    result = find_positions(dimension, nodes)

    for idx, node in enumerate(graph.nodes(data=True)):
        node[1]["layer"] = result[idx]

    edge_styles = {} 
    for u, v, data in graph.edges(data=True):
        edge_styles[(u, v)] = data["weight"]

    print('pos -...')

    pos = nx.multipartite_layout(graph, subset_key='layer')

    # Set up the figure with reduced margin space
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    print('plotting -...')


    nx.draw(graph, pos, node_color='black', with_labels=False, node_size=0.01, edge_color='#A8A8A8', width=0.00025, arrows=False, alpha=0.5)
    plt.tight_layout()
    plt.savefig('mpge.png', dpi=500, bbox_inches='tight')
    #plt.show()
def find_positions(vector, elements):
    cumulative_sum = [0]
    for val in vector:
        cumulative_sum.append(cumulative_sum[-1] + val)

    positions = []

    for element in elements:
        for i in range(len(cumulative_sum) - 1):
            if cumulative_sum[i] <= element < cumulative_sum[i + 1]:
                positions.append(i)
                break
        else:
            positions[element].append(None)

    return positions

def add_all_residual(TG, residual_graphs_out, index_residual_out, dim,w):

    for idx, item in enumerate(residual_graphs_out):
        try:
            g1_w = TG.es['weight']
            offset_left = sum(dim[:index_residual_out[idx]-1])
            off2 = 0
            for i in range(w-1):
                off2 += dim[index_residual_out[idx]+i]
            offset_right= offset_left + off2
            edges_to_add = [(edge.source + offset_left, edge.target + offset_right) for edge in item.es]
            TG.add_edges(edges_to_add)
            weight_residual  = [i for i in item.es['weight']]
            TG.es['weight'] = g1_w + weight_residual
        except:
            pass

    return TG
