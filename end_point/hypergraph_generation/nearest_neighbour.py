import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch import sparse
import time
from .hypergraph import HyperG

def generate_hyper_graph(X,hyper_scales, is_prob=True, with_feature=False):
    
    # X = num_of_peds X feature_dimesion
    # possible_hyper_scales = a tensor representing which are all posible scales are possible
    # Example : X = 7x64
    # hyper_scales = [2,3,5,7,9]

    device = X.get_device()
    batch = X.shape[0]
    n_nodes = X.shape[1]

    # possible_hyper_scales = [2,3,5,7]
    possible_hyper_scales = [value for value in hyper_scales if value <= n_nodes]
    #m_dist = distance between any two nodes in the graph. dimension = 7x7 where m_dist[i][j] = distance between i and j in the hidden feature dimension
    m_dist = torch.cdist(X, X)
    H_sparse_list = []
    # H_non_prob_sparse_list
    weight_combined = torch.tensor([],device=device)
    
    m_neighbors_orig = torch.argsort(m_dist, dim=2)
    m_neighbors_val_orig = torch.gather(m_dist, 2, m_neighbors_orig)

    H = []
    W = []
    edge_idx_start = 0
    for scale in possible_hyper_scales:
        # print("cur_scacle", scale)
        n_edges = n_nodes
        n_neighbors = scale-1
        # argsort => similiar to quicksort
        # m_neighours now contain index such that m_dist is at the sorted place but all small elemnts are towards the left of it and all big elemnts are towards the right of it
        
        # while m_neighours just showed the index value, the m_neighbours_val contains the actual value instead of the index.

        #now slice to take only first m_neighbours
        # m_neighbors = (num_of_peds x m_neigbours)
        # m_neighbors_val = (num_of_peds x m_neigbours)
        m_neighbors = m_neighbors_orig[:, :, :n_neighbors+1]
        m_neighbors_val = m_neighbors_val_orig[ :, :, :n_neighbors+1]
        # print("values_1",m_neighbors)
        # print("values_2",m_neighbors_val)

        # node_idx => dimension now becomes  1d array containing (num_of_pedsxm_neighbous) elements
        node_idx = m_neighbors.reshape(batch,-1)

        # the complex thing finally generates the edge in the format (0,0,0,0,1,1,1,1,2,2,2,2...)
        edge_idx =  torch.arange(start = edge_idx_start,end = edge_idx_start+n_edges).unsqueeze(1).expand(batch, -1, n_neighbors + 1).reshape(batch, -1).to(X.get_device())
        edge_idx_start += n_edges
        if not is_prob:
            values = torch.ones(batch,node_idx.shape[1])
        else:
            avg_dist_2 = torch.mean(m_dist, dim=-1)
            avg_dist  = torch.mean(avg_dist_2, dim = 1)
            m_neighbors_val = m_neighbors_val.reshape(batch,-1)
            # look at the paper https://ieeexplore.ieee.org/abstract/document/9264674
            values = torch.exp(-torch.pow(m_neighbors_val, 2.) / torch.pow(avg_dist, 2.).view(-1,1))

        non_prob_values  = torch.ones(batch,node_idx.shape[1],device=device)

        row = node_idx
        col = edge_idx

        indices = torch.stack((row.unsqueeze(1),col.unsqueeze(1)),dim = 1).squeeze(2)
        # print("shape", non_prob_values.shape)
        # print("shape 2",values.shape)
        # Check for NaN values
        nan_indices = torch.isnan(values)
        values[nan_indices] = 1
        W.append(values)
        H.append(indices)
        # H = torch.sparse_coo_tensor(indices,values,size=(batn_nodes, n_edges), device=device)
        # H_sparse_list.append(H)

        # H_non_prob = torch.sparse_coo_tensor(indices,non_prob_values,size=(n_nodes, n_edges))
        # H_non_prob = H_non_prob.to_dense()
        
        # A_SSA.fill_diagonal_(0)
        # temp = torch.transpose(H.to_dense(),0,1)
        # ssa_combined = torch.matmul(torch.transpose(H_non_prob,0,1),torch.matmul(temp,H_non_prob))
        # w = torch.diagonal(ssa_combined)
        # w /= (scale*(scale))
        # w  =  torch.sum(torch.transpose(H_non_prob,0,1), dim=1)
        # weight_combined = torch.cat((weight_combined,w))

    H = torch.cat(H,dim = -1)
    W = torch.cat(W, dim  = -1)
    return H, W