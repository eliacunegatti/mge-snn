import sys
sys.path.insert(0, '')

import copy
import itertools
import numpy as np
from abc import ABC
from math import floor

#import torch libraries
import torch
import torch.nn as nn

#import graph library
from igraph import *
import networkx as nx

#import local models
from models.resnet import BasicBlock as Basic_Block_resnet
from models.wide_resnet import BasicBlock as Basic_Block_wide_resnet

#import numba optimization
from graph_encoding.numba_utils import optimize_cnn

class GraphEncoding(ABC):
    def __init__(self, model, input_size=None):
        self.model = model        
        self.I = input_size
        self.layers = []
        self.get_residual_counts()
        self.get_layer_type()
        self.W = []
        self.fmap_sizes = []
        self.bdim = []
        self.fmap_sizes_residual = []
        self.bdim_residual = []
        self.feature_map_size()
        
    def get_residual_counts(self):
        self.residual_counts = 0
        conv_list = []
        for _, layer__ in enumerate(self.model.named_modules()):
            if isinstance(layer__[1], Basic_Block_resnet) or isinstance(layer__[1], Basic_Block_wide_resnet):
                #count how many Conv2 in the block
                for _, layer_ in enumerate(layer__[1].named_modules()):
                    if isinstance(layer_[1], nn.Conv2d):
                        if 'conv' in layer_[0].lower() and ('shortcut' not in layer_[0].lower() and 'downsample' not in layer_[0].lower()):
                            conv_list.append(layer_[0])
                        elif 'shortcut' in layer_[0].lower() or 'downsample' in layer_[0].lower():
                            self.residual_counts += 1
        self.window_residual = len(list(set(conv_list)))

    ## store all the model's layers
    def get_layer_type(self): 
        for _, layer__ in enumerate(self.model.named_modules()):
            if isinstance(layer__[1], nn.Conv2d) and 'shortcut' not in layer__[0].lower() and 'downsample' not in layer__[0].lower():
                self.layers.append(tuple([layer__[1], 'conv', layer__[0]]))
            elif isinstance(layer__[1], nn.Linear):
                self.layers.append(tuple([layer__[1], 'linear', layer__[0]]))
            elif isinstance(layer__[1], nn.MaxPool2d) or isinstance(layer__[1], nn.AvgPool2d) or isinstance(layer__[1], nn.AdaptiveAvgPool2d):
                self.layers.append(tuple([layer__[1], 'pooling', layer__[0]]))
            elif isinstance(layer__[1], nn.Conv2d) and 'shortcut' in layer__[0].lower() or 'downsample' in layer__[0].lower():
                self.layers.append(tuple([layer__[1], 'residual', layer__[0]]))

    # given the input size dimension computed the feature map input and output dimension for every layer
    # it also include the dimensionally reduction employed by the pooling layer
    # such input/output dimension are fundamental for the correct graph costruction
  
    def feature_map_size(self):
        if self.I != None:
            I = copy.deepcopy(self.I)
            for idx, l in enumerate(self.layers):
                if l[1] == 'conv':
                    out_size = (floor((I - l[0].kernel_size[0] + (2*l[0].padding[0])) /l[0].stride[0]) +1)
                    I = I + int(2*l[0].padding[0])
                    if int(((I- l[0].kernel_size[0]) / l[0].stride[0]) +1) - (((I- l[0].kernel_size[0]) / l[0].stride[0]) +1) != 0:
                        inn = ((out_size - 1) * l[0].stride[0]) + l[0].kernel_size[0]# - (2*l[0].padding[0])
                        I = inn
                    
                    a = I
                    item = tuple([I*I*self.layers[idx][0].in_channels,  l[0].out_channels * out_size**2])
                    self.bdim.append(item)
                    I = out_size
                    b = I
                    self.fmap_sizes.append(tuple([a, b]))
                    self.W.append(l[0].weight.data)

                elif l[1] == 'pooling':
                    a = I
                    if type(l[0].kernel_size) == int:
                        kernel_size = int(l[0].kernel_size)
                    else:
                        kernel_size = int(l[0].kernel_size[0])
                    if type(l[0].stride) == int:
                        stride = int(l[0].stride)
                    else:
                        stride = int(l[0].stride[0])      
                    if type(l[0].padding) == int:
                        padding = int(l[0].padding)
                    else:
                        padding = int(l[0].padding[0])    
                    I = int(((I - kernel_size + (2*padding)) / stride) + 1)    
                    b =  I
                    self.W.append(None)
                    self.fmap_sizes.append(tuple([a, b]))
                    self.bdim.append((None, None))

                elif l[1] == 'linear':
                    self.fmap_sizes.append(tuple([self.layers[idx][0].in_features,l[0].out_features]))
                    self.W.append(l[0].weight.data)
                    self.bdim.append(tuple([self.layers[idx][0].in_features,l[0].out_features]))

                if l[1] == 'residual':
                    if self.fmap_sizes[idx-1-self.window_residual][1] is not None:
                        D = self.fmap_sizes[idx-1-self.window_residual][1]
                    else:
                        D = self.fmap_sizes[idx-self.window_residual][1]

                    out_size = (floor((D- l[0].kernel_size[0] + (2*l[0].padding[0])) /l[0].stride[0]) +1)
                    item = tuple([D*D*self.layers[idx][0].in_channels,  l[0].out_channels * out_size**2])
                    self.bdim_residual.append(item)
                    self.fmap_sizes_residual.append(tuple([D, out_size]))
                
                    self.W.append(l[0].weight.data)
                    self.bdim.append(None) 
                    self.fmap_sizes.append((None, None))
                else:
                    self.fmap_sizes_residual.append(None)
                    self.bdim_residual.append(None)

        else:
            for idx, l in enumerate(self.layers):
                if l[1] == 'conv' or l[1] == 'residual':
                    self.fmap_sizes.append(tuple([self.layers[idx][0].in_channels,l[0].out_channels]))
                    self.bdim.append(tuple([self.layers[idx][0].in_channels,l[0].out_channels]))
                    self.W.append(l[0].weight.data)
                elif l[1] == 'pooling':
                    self.W.append(None)
                    self.fmap_sizes.append(None)
                    self.bdim.append(None)
                elif l[1] == 'linear':
                    self.W.append(l[0].weight.data)
                    self.fmap_sizes.append(tuple([self.layers[idx][0].in_features,l[0].out_features]))
                    self.bdim.append(tuple([self.layers[idx][0].in_features,l[0].out_features]))

    #linear layer graph construction    
    def get_graph_linear(self, index, mode='networkx'):
        layers_size = list(self.layers[index][0].weight.data.shape)[::-1]
        l1 = [x for x in range(layers_size[0])]
        l2 = [x+1+max(l1) for x in range(layers_size[1])]
        connection = []
        weight = np.transpose(self.W[index].data)
        dim_in, dim_out = weight.shape
        edge_weight = []
        for idx_i in range(dim_in):
            for idx_j in range(dim_out):
                if np.abs(weight[idx_i, idx_j]):
                    connection.append(tuple([l1[idx_i], l2[idx_j]]))
                    edge_weight.append(np.abs(weight[idx_i, idx_j]))


        if mode == 'networkx':
            layers_size = [len(l1), len(l2)]    
            G = nx.DiGraph()
            extents = nx.utils.pairwise(itertools.accumulate((0,) + tuple(layers_size)))
            layers = [range(start, end) for start, end in extents]
            for (i_, layer) in enumerate(layers):
                G.add_nodes_from(list(layer), layer=i_)
            G.add_edges_from(connection)  
        
        elif mode == 'igraph':
            v1 = [False for item in l1]
            v2 = [True for item in l2]
            nodes = v1 + v2
            G = Graph(directed=True)
            vertices = [i for i in range(len(nodes))]
            G.add_vertices(vertices)
            G.vs["type"] = nodes
            G.add_edges(connection)
            edge_weight = np.array(edge_weight, dtype=np.float16)          
            edge_weight = np.abs(edge_weight)      
            G.es["weight"] = edge_weight
        else:
            raise Exception('Not Implemented')
        return G, None


    def add_pooling(self, x, type):
        self.layers.append(tuple([x, type]))



class Unrolled_GE(GraphEncoding):
    """
    Unrolled convolutional graph encoding as in the main paper
    """
    def get_graph(self, index, mode='networkx'):
        if self.layers[index][1] == 'conv' or self.layers[index][1] == 'residual':
            G,_ = self.get_graph_conv_unrolled(index, mode)
        elif self.layers[index][1] == 'linear':
            G,_ = self.get_graph_linear(index, mode)
        else:
            G = None
        return G 
    

    def get_graph_conv_unrolled(self, index, mode='networkx'):
            if self.layers[index][1] == 'conv':
                INPUT_SIZE = self.fmap_sizes[index][0]
            elif self.layers[index][1] == 'residual':
                INPUT_SIZE = self.fmap_sizes_residual[index][0]

            in_channels = self.layers[index][0].in_channels
            out_channels = self.layers[index][0].out_channels
            kernel_size = int(list(self.layers[index][0].kernel_size)[0])
            padding = 0
            stride = int(list(self.layers[index][0].stride)[0])
            INPUT_SIZE_appo = None
            out_size = (floor((INPUT_SIZE - kernel_size + (2*padding)) / stride) +1)

            item = np.array([None,None])
            if padding > 0:
                INPUT_SIZE = INPUT_SIZE + 2*padding
            
            item[0] = INPUT_SIZE*INPUT_SIZE*in_channels
            item[1] = out_channels * out_size**2


            l1 = np.array([x for x in range(item[0])])
            l2 =  np.array([item[0]+x for x in range(item[1])])
            
            l1_split = []
            l2_split = []

            k = 0
            for i in range(in_channels):
                l = [x+k for x in range(int(len(l1)/in_channels))]
                k = max(l) +1
                l1_split.append(l)


            for i in range(out_channels):
                l = [x+k for x in range(out_size**2)]
                k = max(l) +1
                l2_split.append(l)
            

            if INPUT_SIZE_appo == None:
                conv_steps =  int((((INPUT_SIZE -  kernel_size) / stride) + 1)**2)
            else:
                conv_steps =  int((((INPUT_SIZE_appo -  kernel_size) / stride) + 1)**2)
            f = []
            w_original =  self.W[index]
            
            for i in range(out_channels):
                if i == 0:
                    f.append(0)
                else:
                    f.append(f[i-1] + in_channels)

            connection = []
            edge_weight = []
            is_group = None

            if self.layers[index][0].groups != 1 and self.layers[index][0].groups == in_channels:
                is_group = True
            elif self.layers[index][0].groups == 1:
                is_group = False
            elif self.layers[index][0].groups != 1 and self.layers[index][0].groups != in_channels:
                raise Exception('Not Implemented')
            elif self.layers[index][0].groups == 1:
                is_group = False


            l1_split = np.array(l1_split)
            l2_split = np.array(l2_split)

            w_original = w_original.cpu().numpy()
            connection, edge_weight = optimize_cnn(in_channels, out_channels, is_group, w_original, l1_split, l2_split, conv_steps, kernel_size, stride, INPUT_SIZE)
    
            connection = connection[1:]
            edge_weight = edge_weight[1:]
            if mode == 'networkx':
                layers_size = [len(l1), len(l2)]    
                G = nx.DiGraph()
                extents = nx.utils.pairwise(itertools.accumulate((0,) + tuple(layers_size)))
                layers = [range(start, end) for start, end in extents]

                for (i_, layer) in enumerate(layers):
                    G.add_nodes_from(list(layer), layer=i_)
                G.add_weighted_edges_from(connection)  

            elif mode == 'igraph':
                v1 = [False for item in l1]
                v2 = [True for item in l2]
                nodes = v1 + v2
                G = Graph(directed=True)
                vertices = [i for i in range(len(nodes))]
                G.add_vertices(vertices)
                G.vs["type"] = nodes
                G.add_edges(connection)
                edge_weight = np.array(edge_weight, dtype=np.float16)      
                edge_weight = np.abs(edge_weight)      
                G.es["weight"] = edge_weight

            else:
                raise Exception('Not Implemented')
            
            return G, is_group


############ Rolled Encoding Proposed in https://openreview.net/forum?id=uVcDssQff_ #####################
        
class Rolled_GE(GraphEncoding):
    def get_graph(self, index, mode='networkx'):
        if self.layers[index][1] == 'conv' or self.layers[index][1] == 'residual':
            G,_ = self.get_graph_conv_rolled(index, mode)
        elif self.layers[index][1] == 'linear':
            G,_ = self.get_graph_linear(index, mode)
        else:
            G = None
        return G         

    def get_graph_conv_rolled(self, index, mode='networkx'):
        weight = self.W[index]
        nx_new = nx.DiGraph()
        if len(weight.shape) == 4:  # c_out x c_in x kH x kw
            c_out = weight.shape[0]
            weight = weight.reshape(c_out, -1)
            weight = np.transpose(weight)
            assert weight.shape[1] == c_out
        else:
            weight = np.transpose(weight)
        dim_in, dim_out = weight.shape

        edge_weight = []

        for i in range(dim_in):
            nx_new.add_node(i, layer=0)
            for j in range(dim_out):
                idx_in = i
                idx_out = j + dim_in
                nx_new.add_node(idx_out, layer=1)
                edge_w = np.abs(weight[i, j])
                if edge_w > 0:
                    nx_new.add_edges_from([(idx_in, idx_out)])
                    edge_weight.append(edge_w)


        if mode == 'networkx':
            return nx_new, False
        else:
            v1 = [False for item in range(dim_in)]
            v2 = [True for item in range(dim_out)]
            nodes = v1 + v2
            G = Graph(directed=True)
            vertices = [i for i in range(len(nodes))]
            G.add_vertices(vertices)
            G.vs["type"] = nodes
            G.add_edges(nx_new.edges())
            edge_weight = np.array(edge_weight, dtype=np.float16)   
            edge_weight = np.abs(edge_weight)      
            G.es["weight"] = edge_weight

            return G, False
        


############ Rolled_Channel_Encoding Encoding Proposed in https://openaccess.thecvf.com/content_ECCV_2018/papers/Ameya_Prabhu_Deep_Expander_Networks_ECCV_2018_paper.pdf   #####################


class Rolled_Channel_GE(GraphEncoding):
    def get_graph(self, index, mode='networkx'):
        if self.layers[index][1] == 'conv' or self.layers[index][1] == 'residual':
            G,_ = self.get_graph_conv_rolled_channel(index, mode)
        elif self.layers[index][1] == 'linear':
            G,_ = self.get_graph_linear(index, mode)
        else:
            G = None
        return G 

    def get_graph_conv_rolled_channel(self, index, mode='networkx'):
        in_channels = self.layers[index][0].in_channels
        out_channels = self.layers[index][0].out_channels


        L = in_channels 
        R = out_channels
        item = [None,None]

        item[0] = L
        item[1] = R

        l1 = [x for x in range(item[0])]
        l2 = [item[0]+x for x in range(item[1])]
        l1_split = []

        
        k = 0
        for i in range(in_channels):
            l = [x+k for x in range(int(len(l1)/in_channels))]
            k = max(l) +1
            l1_split.append(l)


        connection = []

        is_group = None

        if self.layers[index][0].groups != 1 and self.layers[index][0].groups == in_channels:
            is_group = True
        elif self.layers[index][0].groups == 1:
            is_group = False
        elif self.layers[index][0].groups != 1 and self.layers[index][0].groups != in_channels:
            raise Exception('Not Implemented')
        elif self.layers[index][0].groups == 1:
            is_group = False


        w_original =  self.W[index]
        edge_weight = []
        for j in (range(in_channels)):
            for c in (range(out_channels)):
                if is_group:
                    w = w_original[j][0]
                    l = l1_split[j][0]
                    r = l2[j]
                else:
                    w = w_original[c][j]
                    l = l1_split[j][0]
                    r = l2[c]

                w = [float(x) for x in list(torch.flatten(np.transpose(w)))]
                if w.count(0) != len(w):
                    connection.append(tuple([l, r]))
                    edge_weight.append(torch.norm(torch.tensor(w), p=2))
                else:
                    pass
                if is_group:
                    break

        if mode == 'networkx':
            layers_size = [len(l1), len(l2)]    
            G = nx.DiGraph()
            extents = nx.utils.pairwise(itertools.accumulate((0,) + tuple(layers_size)))
            layers = [range(start, end) for start, end in extents]

            for (i_, layer) in enumerate(layers):
                G.add_nodes_from(list(layer), layer=i_)
            G.add_weighted_edges_from(connection)  

        elif mode == 'igraph':
            v1 = [False for item in l1]
            v2 = [True for item in l2]
            nodes = v1 + v2
            G = Graph(directed=True)
            vertices = [i for i in range(len(nodes))]
            G.add_vertices(vertices)
            G.vs["type"] = nodes
            G.add_edges(connection)
            edge_weight = np.array(edge_weight, dtype=np.float16)    
            edge_weight = np.abs(edge_weight)      
            G.es["weight"] = edge_weight
        
        else:
            raise Exception('Not Implemented')
        return G, is_group


