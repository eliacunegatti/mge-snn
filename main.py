import os
max_threads = 5
# Set the OMP_NUM_THREADS environment variable
os.environ["OMP_NUM_THREADS"] = str(max_threads) 
os.environ["OPENBLAS_NUM_THREADS"] =  str(max_threads)  
os.environ["MKL_NUM_THREADS"] =  str(max_threads) 
os.environ["VECLIB_MAXIMUM_THREADS"] =  str(max_threads)  
os.environ["NUMEXPR_NUM_THREADS"] =  str(max_threads) 

import argparse

import progressbar
import torch
import torch.nn.functional as F
#base libraries
import copy
import math
import torch
import random
import argparse
import pandas as pd
import progressbar
#graph libraries


#graph analysis libraries

## local functions
from graph_encoding.numba_utils import *
from graph_encoding.graph_encoding import *
from multipartite_encoding import *

import scipy as sp


from utils.datasets import get_dataset
from utils.models import get_model
from utils.masking import apply_prune_mask, count_paramaters, mask_gradient, mask_bias
from utils.init import init_weights, mask_bias
from utils.hyperparameters import get_optimizer, get_scheduler
from utils.pai.snip import SNIP
from utils.pai.grasp import GraSP
from utils.pai.synflow import *
import utils.pai.Prospr.prospr as psr
from utils.pai.Prospr.utils import pruning_filter_factory

from utils.pai.Rand import Rand
from utils.sparselearning.core import Masking

from ramanunjan_metric import *
##### Train and Test Loop #####

def train(model, trainloader, optimizer, e, device):
    model.train()
    running_loss = 0
    losses = []
    n_correct = 0
    num_samples = 0
    for images, labels in progressbar.progressbar(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, labels)
        
        _, predictions = output.max(1)
        n_correct += (predictions == labels).sum()
        num_samples += predictions.size(0)

        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()

    train_acc = (100 * float(n_correct) / float(num_samples))
    train_loss = (float(running_loss/len(trainloader)))
    print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

    return train_acc, train_loss


def test(model, testloader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        all_count = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_count += labels.size(0)
            correct += (predicted == labels).sum().item()

    return float(correct/all_count)


##### Eigenvalues #####

def get_eig_values(matrix: np.array, k: int = 3):
    """
    get the real eig of a square matrix
    for bi-graph, the third largest eig denotes connectivity
    """

    adj_eigh_val, _ = sp.sparse.linalg.eigsh(matrix, k=3, which='LM')
    abs_eig = [abs(i) for i in adj_eigh_val]
    abs_eig.sort(reverse=True)
    return abs_eig


def get_eig_values_laplacian(matrix: np.array, k: int = 3):
    """
    get the real eig of a square matrix
    for bi-graph, the third largest eig denotes connectivity
    """

    adj_eigh_val, _ = sp.sparse.linalg.eigsh(matrix, k=2, which='LM')
    abs_eig = [i for i in adj_eigh_val]
    abs_eig.sort(reverse=True)
    return abs_eig




##### Rolled Channel Layer-by-Layer Metric #####
def process_graph_ramanunjan(graph1, idx_layer, data_final, *kwargs):

    G = graph1.get_graph(idx_layer, mode='igraph')
    if G is not None:
        l1, r1 = get_ids(G)
        edges = G.ecount()
        nodes = G.vcount()
        layer_sparsity = float(torch.count_nonzero(graph1.W[idx_layer]) / graph1.W[idx_layer].numel())
        no_weights = float(torch.count_nonzero(graph1.W[idx_layer]))
        no_params = float(graph1.W[idx_layer].numel())

        iterative_results = iterative_mean_spectral_gap(G, len(l1), len(r1))[0]



        degree = G.degree()
        G, dim_in, dim_out = filter_zero_degree_general(G, len(l1))
        degree = G.degree()

        if len(degree[dim_in:]) > 0 and len(degree[0:dim_in]) > 0:
            d_avg_l = np.mean(degree[0:dim_in])
            d_avg_r = np.mean(degree[dim_in:])
            weights = np.array(G.es['weight'], dtype=np.float32)
            edges_ = np.array(G.get_edgelist())
            num_vertices = G.vcount()
            layer = {'dim_in': dim_in, 'dim_out': dim_out}

            row_indices, col_indices, data = igraph_edges_to_sparse_matrix_weighted(edges_, weights, mode='ALL')
            adj_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))
            _, delta_r_direct_both = None,None
            if adj_matrix.shape[0] > 3:
                
                G = G.as_undirected(combine_edges='first')
                G.to_undirected()
                G.to_directed()

                _ , delta_r_direct_both = get_all_metrics(G, layer, degree, dim_in, d_avg_l, d_avg_r)

        else:
            _ , delta_r_direct_both = None, None

        data_final[f'{kwargs[0]}-{kwargs[1]}-{args.dataset}-{args.model}-seed-{kwargs[2]}-{idx_layer}'] = [idx_layer,
                                                                                        graph1.layers[idx_layer][2],
                                                                                        kwargs[1], kwargs[0], kwargs[2],
                                                                                        args.model, args.dataset, 
                                                                                        iterative_results[0],
                                                                                        iterative_results[1],
                                                                                        delta_r_direct_both,
                                                                                        edges, nodes,
                                                                                        layer_sparsity, no_weights, no_params]                                                          

        return data_final

def process_graph_topometrics(graph1, idx_layer, data_final, *kwargs):

    G = graph1.get_graph(idx_layer, mode='igraph')
    if G is not None:
        bridge = len(G.bridges()) 
        articulation_points = len(G.articulation_points()) 
        motifs_count = G.motifs_randesu_estimate(size=4, sample=int(0.001*len(G.vs))) #okat
        nei_one_hop =  np.mean(G.neighborhood_size(order=1, mode='out')) 
        strenght_all = np.mean(G.strength(weights='weight', mode='all')) 
        

        coreness_all = np.mean(G.coreness(mode='all')) 

        G = G.as_undirected(combine_edges='first')
        G.to_directed()

        strong_connected = G.connected_components(mode='strong')
        strong_clusters = len(strong_connected) 
        mean_strong = np.mean([len(i) for i in strong_connected]) 


        #please not that train is set to False the model is not trained hence the expansion propreties are computed over
        #the random initialized weights rather than the trained weights
        row_indices, col_indices, data = igraph_edges_to_sparse_matrix_weighted( np.array(G.get_edgelist()), np.array(G.es['weight'], dtype=np.float32), mode='ALL')
        sparse_matrix_weighted = csc_matrix((data, (row_indices, col_indices)), shape=(int(G.vcount()), int(G.vcount())))         

        m_eig_vals = get_eig_values(sparse_matrix_weighted)
        t2_m, t1_m = m_eig_vals[-1], m_eig_vals[0]
        spectral_gap_weighted_undirected = (t1_m - t2_m)

        laplacian_matrix = csc_matrix(laplacian_from_adjacency(sparse_matrix_weighted))
        m_eig_vals = get_eig_values_laplacian(laplacian_matrix)
        t2_m, t1_m = m_eig_vals[1], m_eig_vals[0]
        spectral_radius_laplacian_weighted_undirected = t1_m


        data_final[f'{kwargs[0]}-{kwargs[1]}-{args.dataset}-{args.model}-seed-{kwargs[2]}-{idx_layer}'] = [idx_layer,
                                                                                        graph1.layers[idx_layer][2],
                                                                                        kwargs[1], kwargs[0], kwargs[2],
                                                                                        args.model, args.dataset, 
                                                                                        bridge, articulation_points, motifs_count, nei_one_hop, strenght_all, coreness_all, strong_clusters, mean_strong, spectral_gap_weighted_undirected, spectral_radius_laplacian_weighted_undirected]                                                      

        return data_final

def apply_pruning(args, model, trainloader, device, classes):
    if args.pruning == 'snip':
        keep_masks = SNIP(model, 1-args.sparsity, trainloader, device)
    elif args.pruning == 'grasp':
        keep_masks = GraSP(model,1-args.sparsity, trainloader, num_classes=classes, device='cuda')
        model.to('cuda')
    elif args.pruning == 'synflow':
        input_shape = list(trainloader.dataset.__getitem__(0)[0][:].shape)
        model_copy = copy.deepcopy(model)
        model_copy.state_dict(model.state_dict())
        pruner = Synflow(model_copy, device, input_shape=input_shape)
        pruner.prune(amount=args.sparsity)
        keep_masks = []
        for item in pruner.mask:
            if len(item.shape) > 1:
                print(item.shape, torch.count_nonzero(item)/item.numel())
                keep_masks.append(item)  
        del model_copy
        del pruner
        torch.cuda.empty_cache() 
    elif args.pruning == 'prospr':
        filter_fn = pruning_filter_factory(classes, False)
        _, keep_masks = psr.prune(
            model,
            prune_ratio=args.sparsity,
            dataloader=args.dataset,
            filter_fn=filter_fn,
            num_steps=3,
            inner_lr=0.5, 
            inner_momentum=0.9,
        )
    elif args.pruning == 'uniform':
        model_copy = copy.deepcopy(model)
        model_copy.state_dict(model.state_dict())
        input_shape = list(trainloader.dataset.__getitem__(0)[0][:].shape)
        pruner  = Rand(model_copy, device, input_shape)
        pruner.prune(args.sparsity)
        syn_mask =  pruner.get_prune_score()
        del model_copy
        del pruner
        index = 0
        keep_masks = []
        for i in range(len(syn_mask)):
            if len(list(syn_mask[i].shape)) > 1:
                size = tuple(syn_mask[i].shape)
                flat_syn_mask = list(torch.flatten(syn_mask[i]))
                flat_syn_mask =  [1 if float(x) != 0.0 else 0 for x in flat_syn_mask]
                unflatten = nn.Unflatten(-1, size)
                keep_masks.append(unflatten(torch.Tensor(flat_syn_mask)))
                print(keep_masks[index].shape)
                index +=1   
    elif args.pruning == 'er':
        mask = Masking(None, None,)
        mask.add_module(model, density=float(1-args.sparsity),device=device, sparse_init='ER')

        keep_masks = [] 
        for _, value in mask.masks.items():
            keep_masks.append(value)
    elif args.pruning == 'erk':
        mask = Masking(None, None,)
        mask.add_module(model, density=float(1-args.sparsity), device=device,sparse_init='ERK')
        keep_masks = [] 
        for _, value in mask.masks.items():
            keep_masks.append(value)
    elif args.pruning == 'dense':
        pass
    
    else:
        raise ValueError('Pruning algorithm not implemented')

    return keep_masks


def training_and_test(model, trainloader, testloader, device, args):
    optimizer , epochs = get_optimizer(model, args.dataset, args.model)
    lr_scheduler = get_scheduler(optimizer, args.dataset, args.model)

    loss_list = []
    epochs = 2
    for epoch in range(epochs):
        _, train_loss = train(model, trainloader, optimizer, epoch+1, device)
        loss_list.append(train_loss)
        if lr_scheduler is not None:
            lr_scheduler.step()

    print('Test Accuracy:', test(model, testloader, device))

def get_encoding(args, model, input_size):

    graph, dimension = None, None
    if args.graph_metrics == 'ramanunjan':
        if args.encoding == 'rolled':
            encoding = Rolled_GE(model)
        elif args.encoding == 'rolled-channel':
            encoding = Rolled_Channel_GE(model)
        else:
            Exception('Ramanunjan metrics not  implemented for unrolled graph encoding')
    elif args.graph_metrics == 'topometrics':
        if args.encoding == 'unrolled':
            encoding = Unrolled_GE(model=model, input_size=input_size)
            graph, dimension = get_multipartite_graph(encoding)
        elif args.encoding == 'rolled-channel':
            encoding = Rolled_Channel_GE(model)
        elif args.encoding == 'rolled':
            encoding = Rolled_GE(model)
        else:
            Exception('Graph encoding not implemented')

    return graph, encoding, dimension

def main(args):

    if args.graph_metrics == 'ramanunjan' and args.encoding == 'unrolled':  raise Exception('Ramanunjan metrics not implemented for unrolled graph encoding')
    load_numba()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainloader, testloader, input_size, input_channels, classes = get_dataset(args.dataset)  
    device =  args.device

    model = get_model(args.model, input_channels, classes).to(device)
    model.apply(init_weights)

    keep_masks = apply_pruning(args, model, trainloader, device, classes)

    apply_prune_mask(model, keep_masks, device)
    mask_gradient(model, keep_masks)
    model.apply(mask_bias)
    count_paramaters(model)

    if args.training:
        training_and_test(model, trainloader, testloader, device, args)
    multipartite_graph, encoding, dimension = get_encoding(args, model, input_size)
    
    del model
    torch.cuda.empty_cache()

    print('Graph Encoding:', args.encoding)
    print('Graph Metrics:', args.graph_metrics)


    if args.graph_metrics == 'topometrics' and args.encoding == 'unrolled':
        cut_edges = len(multipartite_graph.bridges()) 
        cut_nodes = len(multipartite_graph.articulation_points()) 
        motifs_count = multipartite_graph.motifs_randesu_estimate(size=4, sample=int(0.001*len(multipartite_graph.vs))) #okat
        nei_one_hop =  np.mean(multipartite_graph.neighborhood_size(order=1, mode='out')) 
        nei_two_hop =  np.mean(multipartite_graph.neighborhood_size(order=2, mode='out')) 
        strenght_all = np.mean(multipartite_graph.strength(weights='weight', mode='all')) 
        
        sink_nodes = len([node.index for node in multipartite_graph.vs if node.outdegree() == 0 and node.indegree() !=0]) - dimension[-1] 
        source_nodes = len([node.index for node in multipartite_graph.vs if node.outdegree() != 0 and node.indegree() ==0]) - dimension[0] 
        disconnected_nodes =  len([node.index for node in multipartite_graph.vs if node.outdegree() == 0 and node.indegree() ==0]) 
        lost_out_edges = sum([node.indegree() for node in multipartite_graph.vs if node.outdegree() == 0 and node.indegree() !=0 and node.index < int(multipartite_graph.vcount()-1) - dimension[-1]]) 
        lost_in_edges = sum([node.outdegree() for node in multipartite_graph.vs if node.outdegree() != 0 and node.indegree() ==0 and node.index > dimension[0]]) 

        coreness_all = np.mean(multipartite_graph.coreness(mode='all')) 

        multipartite_graph = multipartite_graph.as_undirected(combine_edges='first')
        multipartite_graph.to_directed()

        strong_connected = multipartite_graph.connected_components(mode='strong')
        strong_clusters = len(strong_connected) 
        mean_strong = np.mean([len(i) for i in strong_connected]) 


        #please not that train is set to False the model is not trained hence the expansion propreties are computed over
        #the random initialized weights rather than the trained weights
        row_indices, col_indices, data = igraph_edges_to_sparse_matrix_weighted( np.array(multipartite_graph.get_edgelist()), np.array(multipartite_graph.es['weight'], dtype=np.float32), mode='ALL')
        sparse_matrix_weighted = csc_matrix((data, (row_indices, col_indices)), shape=(int(multipartite_graph.vcount()), int(multipartite_graph.vcount())))         

        m_eig_vals = get_eig_values(sparse_matrix_weighted)
        t2_m, t1_m = m_eig_vals[-1], m_eig_vals[0]
        spectral_gap_weighted_undirected = (t1_m - t2_m)

        laplacian_matrix = csc_matrix(laplacian_from_adjacency(sparse_matrix_weighted))
        m_eig_vals = get_eig_values_laplacian(laplacian_matrix)
        t2_m, t1_m = m_eig_vals[1], m_eig_vals[0]
        spectral_radius_laplacian_weighted_undirected = t1_m


        print('################## Selected Combination ##################')
        print('Pruning Algorithm:', args.pruning, 'Sparsity:', args.sparsity, 'Model:', args.model, 'Dataset:', args.dataset)
        print('################## Topometric Metrics ##################')
        print('Local Connectivity:', '\nSink Nodes', sink_nodes, '\nSource Nodes', source_nodes, '\nDisconnected Nodes', disconnected_nodes, '\nNo. removable ougraphoing connections', lost_out_edges, '\nNo. removable incoming connections', lost_in_edges)
        print('-'*15)
        print('Neighborhood Connectivity:', '\nOne Hop', nei_one_hop, '\nTwo Hop', nei_two_hop, '\nMotifs', motifs_count)
        print('-'*15)
        print('Strength Connectivity:', '\nStrength', strenght_all, '\nk-core', coreness_all)
        print('-'*15)
        print('Global Connectivity:', '\ncut-edges', cut_edges, '\ncut-nodes', cut_nodes,'\nConnected Components', strong_clusters, '\nMean Strongly Connected Components', mean_strong)
        print('-'*15)
        print('Expansion (*):', '\nSpectral Gap', spectral_gap_weighted_undirected, '\nSpectral Radius', spectral_radius_laplacian_weighted_undirected)

        #save to csv
        data_final = {'sparsity': [args.sparsity], 'pruning': [args.pruning], 'seed': [args.seed], 'model': [args.model], 'dataset': [args.dataset], 'cut-edges': [cut_edges], 'cut-nodes': [cut_nodes], 'motifs': [motifs_count], '1-hop': [nei_one_hop], '2-hop': [nei_two_hop], 'strenght': [strenght_all], 'k-core': [coreness_all], 'connected_components': [strong_clusters], 'avg_connected_components': [mean_strong], 'spectral-gap': [spectral_gap_weighted_undirected], 'spectral-radius': [spectral_radius_laplacian_weighted_undirected]}
        df = pd.DataFrame(data_final)
        df["encoding"] = [args.encoding for i in range(len(df))]

        if os.path.isfile('topometrics_multipartite.csv'):
            df.to_csv('topometrics_multipartite.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('topometrics_multipartite.csv', mode='w', header=True, index=False)

    elif args.graph_metrics == 'topometrics' and (args.encoding == 'rolled-channel' or args.encoding == 'rolled'):
        no_layers = len(encoding.layers)
        data_final = {}
        for idx_layer in range(0, no_layers):
            if encoding.layers[idx_layer][1] == 'residual' or encoding.layers[idx_layer][1] == 'conv' or encoding.layers[idx_layer][1] == 'linear':
                data_final_ = process_graph_topometrics(encoding, idx_layer, data_final, args.pruning, args.sparsity, args.seed)
        df = pd.DataFrame.from_dict(data_final_, orient='index', columns=['layer_id', 'layer_name', 'sparsity', 'pruning_algorithm', 'seed', 'model', 'dataset', 'cut-edges', 'cut-nodes', 'motifs', '1-hop', 'strenght', 'k-core', 'connected_components', 'avg_connected_components', 'spectral-gap', 'spectral-radius'])
        df["encoding"] = [args.encoding for i in range(len(df))]
        if os.path.isfile('topometrics_bipartite.csv'):
            df.to_csv('topometrics_bipartite.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('topometrics_bipartite.csv', mode='w', header=True, index=False)

    elif args.graph_metrics == 'ramanunjan':
        no_layers = len(encoding.layers)
        data_final = {}
        for idx_layer in range(0, no_layers):
            if encoding.layers[idx_layer][1] == 'residual' or encoding.layers[idx_layer][1] == 'conv' or encoding.layers[idx_layer][1] == 'linear':
                data_final_ = process_graph_ramanunjan(encoding, idx_layer, data_final, args.pruning, args.sparsity, args.seed)
        df = pd.DataFrame.from_dict(data_final_, orient='index', columns=['layer_id', 'layer_name', 'sparsity', 'args.pruning_algorithm', 'seed', 'model', 'dataset', 'iterative_spectral_gap','imdb', 'delta_r', 'edges', 'nodes', 'layer_sparsity', 'no_weights', 'no_params'])
        df["encoding"] = [args.encoding for i in range(len(df))]
        if os.path.isfile('ramanunjan.csv'):
            df.to_csv('ramanunjan.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('ramanunjan.csv', mode='w', header=True,  index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Required Arguments')
    parser.add_argument('--model', help='Model', default='Resnet-20',choices=['CONV-6','Resnet-20', 'Resnet-32','Wide-Resnet-28-2'])
    parser.add_argument('--dataset', help='Dataset',  default='CIFAR-10', choices=['CIFAR-10','CIFAR-100','tinyimagenet'])
    parser.add_argument('--sparsity', help='Sparsity', default=0.98, type=float)
    parser.add_argument('--pruning', help='Pruning Algorithm', default='snip',choices=['snip',  'synflow', 'grasp', 'prospr','uniform',  'er', 'erk'], required=False)
    parser.add_argument('--seed', help='Seed', default=0, type=int)
    parser.add_argument('--encoding', help='Graph Encoding', default='unrolled',choices=['rolled','rolled-channel', 'unrolled'])
    parser.add_argument('--device', help='Device', default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--graph_metrics', help='Graph Metrics', default='topometrics', choices=['topometrics', 'ramanunjan'])
    parser.add_argument('--training', help='Training', default=False, type=bool)
    args = parser.parse_args()
    print('Arguments:', args)
    main(args)