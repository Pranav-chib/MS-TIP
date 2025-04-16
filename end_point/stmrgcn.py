import torch
import torch.nn as nn
from .dropedge import drop_edge
from .normalizer import normalized_adjacency_tilde_matrix
import torch.nn.functional as Func
from torch_geometric.nn import HypergraphConv
from torch.utils.data import Dataset

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
    
class SequentialSceneAttention(nn.Module):
    def __init__(self,attn_L=196,attn_D=512,ATTN_D_DOWN=16,bottleneck_dim=4,embedding_dim=10):
        super(SequentialSceneAttention, self).__init__()

        self.L = attn_L  
        self.D = attn_D  
        self.D_down = ATTN_D_DOWN  
        self.bottleneck_dim = bottleneck_dim  
        self.embedding_dim = embedding_dim   

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)    
        self.pre_att_proj = nn.Linear(self.D, self.D_down)       

        mlp_pre_dim = self.embedding_dim + self.D_down  
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)  

        self.attn = nn.Linear(self.L*self.bottleneck_dim, self.L)    

    def forward(self, vgg, end_pos):

        npeds = end_pos.size(1)    
        end_pos = end_pos[0, :, :]     
        curr_rel_embedding = self.spatial_embedding(end_pos)  
        curr_rel_embedding = curr_rel_embedding.view(-1, 1, self.embedding_dim).repeat(1, self.L, 1)  
        vgg=vgg.repeat(npeds,1,1,1)     
        vgg = vgg.view(-1, self.D)   
        features_proj = self.pre_att_proj(vgg)       
        features_proj = features_proj.view(-1, self.L, self.D_down)  

        mlp_h_input = torch.cat([features_proj, curr_rel_embedding], dim=2) 
        attn_h = self.mlp_pre_attn(mlp_h_input.view(-1, self.embedding_dim+self.D_down))  
        attn_h = attn_h.view(npeds, self.L, self.bottleneck_dim)  

        attn_w = Func.softmax(self.attn(attn_h.view(npeds, -1)), dim=1) 
        attn_w = attn_w.view(npeds, self.L, 1)     

        sequential_scene_attention = torch.sum(attn_h * attn_w, dim=1)     
        return sequential_scene_attention 

class HyperGraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention = False,
                 heads=1):
        super(HyperGraphConv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.hyper_conv = HypergraphConv(in_channels,out_channels,use_attention=use_attention)

        # self.hyper_conv = nn.ModuleList()
        # for i in range(8):
        #     self.hyper_conv.append(HypergraphConv(in_channels,out_channels,use_attention=use_attention))

    def forward(self,x,H, sequential_scene_attention, W):
        # x.shape = 1x2x8x3

        batches = x.shape[0]
        features = x.shape[1]
        obs_len = x.shape[2]
        num_of_peds = x.shape[3]


        # x.shape 1 x time x agents x corrdinates
        x = x.permute(0,2,3,1)  
        
        if sequential_scene_attention.shape[0] != batches:
            sequential_scene_attention = sequential_scene_attention.repeat(batches,1,1,1)

        x = torch.cat((x,sequential_scene_attention), dim = 3)
        
        # unified_input=unified_input.view(1,obs_len,num_of_peds,-1)
        x = x.view(batches, obs_len, num_of_peds, -1)

        final_node_feature  = torch.empty(0,obs_len,num_of_peds,self.out_channels,device = x.get_device())
        for batch in range(0, batches):
            # cur_x will have shape of 8x3x2
            cur_x = x[batch,:,:,:]
            cur_h = H[batch]
            cur_w = W[batch]
            cur_node_features = torch.empty(0,num_of_peds,self.out_channels, device = x.get_device())
            for i in range(0,obs_len):
                current_embeddings = cur_x[i]
                current_hyperedge_indicies = cur_h[i]
                current_weights = cur_w[i].detach()
                current_weights = torch.where(current_weights < 0.001, torch.tensor(0.0, device = x.get_device()), current_weights)
                # print(current_weights)
                # print("here")
                ped_nodes_features = self.hyper_conv(x = current_embeddings, hyperedge_index = current_hyperedge_indicies)
                # print(ped_nodes_features)
                cur_node_features = torch.cat((cur_node_features, ped_nodes_features.unsqueeze(0)),dim = 0)
            cur_node_features = cur_node_features.unsqueeze(0)
            final_node_feature = torch.cat((final_node_feature,cur_node_features))

        final_node_feature = final_node_feature.reshape(batches,self.out_channels,obs_len, num_of_peds)
        return final_node_feature.contiguous()

class MultiRelationalGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True, relation=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.relation = relation
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * relation, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A, sequential_scene_attention):
        assert A.size(0) == x.size(0)
        assert A.size(1) == self.relation
        assert A.size(2) == self.kernel_size


        B = x.size(0)
        T = x.size(2)
        N = x.size(3)

        # x.shape 1 x time x agents x corrdinates
        x = x.permute(0,2,3,1)  
        
        if sequential_scene_attention.shape[0] != B:
            sequential_scene_attention = sequential_scene_attention.repeat(B,1,1,1)

        x = torch.cat((x,sequential_scene_attention), dim = 3)

        unified_graph=x.view(B,T,N,-1)

        #unified_graph = 1 x corrdinates x time x agents
        unified_graph=unified_graph.permute(0,3,1,2)
        x = self.conv(unified_graph)
        x = x.view(x.size(0), self.relation, self.out_channels, x.size(-2), x.size(-1))
        x = torch.einsum('nrtwv,nrctv->nctw', normalized_adjacency_tilde_matrix(drop_edge(A, 0.8, self.training)), x)
        return x.contiguous(), A

class st_mrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=True, stride=1, dropout=0, residual=True, relation=2):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.relation = relation
        self.prelu = nn.PReLU()
        self.gcn = HyperGraphConv(in_channels, out_channels)
        self.scene_att=SequentialSceneAttention()
        self.tcn = nn.Sequential(nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.Dropout(dropout, inplace=True),)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(2, out_channels, kernel_size=1, stride=(stride, 1)),)

    def forward(self, x, H, vgg, W):
        # print("X shape", x.shape)
        # X.shape 1x2x8x7
        # X.shape batch x cooridnates x time x agents

        coordinates=x[:,:,-1,:]
        T=x.size(2)
        coordinates=coordinates.permute(0,2,1)  
        sequential_scene_attention=self.scene_att(vgg,coordinates)     
        sequential_scene_attention=sequential_scene_attention.unsqueeze(0)   
        sequential_scene_attention=sequential_scene_attention.unsqueeze(1)   
        sequential_scene_attention=sequential_scene_attention.repeat(1,T,1,1)
        

        res = self.residual(x)
        x = self.gcn(x, H, sequential_scene_attention, W)
        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x

class st_mrgcn_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=True, stride=1, dropout=0, residual=True, relation=2):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.relation = relation
        self.prelu = nn.PReLU()
        self.gcn = MultiRelationalGCN(in_channels, out_channels, kernel_size[1], relation=self.relation)
        self.tcn = nn.Sequential(nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.Dropout(dropout, inplace=True),)
        self.scene_att=SequentialSceneAttention()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(2, out_channels, kernel_size=1, stride=(stride, 1)),)

    def forward(self, x, A, vgg):
        
        coordinates=x[:,:,-1,:]
        T=x.size(2)
        coordinates=coordinates.permute(0,2,1)  
        sequential_scene_attention=self.scene_att(vgg,coordinates)     
        sequential_scene_attention=sequential_scene_attention.unsqueeze(0)   
        sequential_scene_attention=sequential_scene_attention.unsqueeze(1)   
        sequential_scene_attention=sequential_scene_attention.repeat(1,T,1,1)

        res = self.residual(x)
        x, A = self.gcn(x, A, sequential_scene_attention)
        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class epcnn(nn.Module):
    def __init__(self, obs_seq_len, pred_seq_len, in_channels, out_channels, n_tpcn=1, c_ksize=3, n_cpcn=1, t_ksize=3, dropout=0, residual=True):
        super().__init__()

        # NTCV
        self.tpcns = nn.ModuleList()
        for i in range(0, n_tpcn - 1):
            self.tpcns.append(nn.Sequential(nn.Conv2d(obs_seq_len, obs_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True), ))
        self.tpcns.append(nn.Sequential(nn.Conv2d(obs_seq_len, pred_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True), ))

        # NCTV
        self.cpcns = nn.ModuleList()
        for i in range(0, n_cpcn - 1):
            self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True), ))
        self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True), ))

        if not residual:
            self.residual = lambda x: 0
        elif obs_seq_len == pred_seq_len and in_channels == out_channels:
            self.residual = lambda x: x
        elif obs_seq_len == pred_seq_len:
            self.rescconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),)
            self.residual = lambda x: self.rescconv(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()
        elif in_channels == out_channels:
            self.restconv = nn.Sequential(nn.Conv2d(obs_seq_len, pred_seq_len, kernel_size=1),)
            self.residual = lambda x: self.restconv(x)
        else:
            self.rescconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),)
            self.restconv = nn.Sequential(nn.Conv2d(obs_seq_len, pred_seq_len, kernel_size=1),)
            self.residual = lambda x: self.rescconv(self.restconv(x).permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()

    def forward(self, x):
        # residual
        res = self.residual(x)

        # time-wise
        for i in range(len(self.tpcns)):
            x = self.tpcns[i](x)

        # channel-wise
        x = x.permute(0, 2, 1, 3).contiguous()
        for i in range(len(self.cpcns)):
            x = self.cpcns[i](x)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x + res


class trcnn(nn.Module):
    def __init__(self, total_seq_len, pred_seq_len, in_channels, out_channels, n_tpcn=1, c_ksize=3, n_cpcn=1, t_ksize=3, dropout=0, residual=True):
        super().__init__()

        # NTCV
        self.tpcns = nn.ModuleList()
        for i in range(0, n_tpcn-1):
            self.tpcns.append(nn.Sequential(nn.Conv2d(total_seq_len, total_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True),))
        self.tpcns.append(nn.Sequential(nn.Conv2d(total_seq_len, pred_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True),))

        # NCTV
        self.cpcns = nn.ModuleList()
        for i in range(0, n_cpcn-1):
            self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True),))
        self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True),))

        if not residual:
            self.residual = lambda x: 0
        elif total_seq_len == pred_seq_len:
            self.residual = lambda x: x
        else:
            k_size = total_seq_len - pred_seq_len + 1
            self.resconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(k_size, 1)),)
            self.residual = lambda x: self.resconv(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()

    def forward(self, x):
        # residual
        res = self.residual(x)

        # time-wise
        for i in range(len(self.tpcns)):
            x = self.tpcns[i](x)

        # channel-wise
        x = x.permute(0, 2, 1, 3).contiguous()
        for i in range(len(self.cpcns)):
            x = self.cpcns[i](x)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x + res
