
import torch
import torch.nn as nn
import math
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.nn.functional as F


class SpiralConv(nn.Module):
    def __init__(self, num_pts, in_c, spiral_size,out_c,activation='elu',bias=True):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self,x, t_vertex, spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()
  
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]


        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device=x.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat
    

class chebyshevConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self,  num_pts, in_features, kernal_size, out_features, activation='elu', bias=True):
        super(chebyshevConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = kernal_size
        self.weight = nn.Parameter(torch.FloatTensor(in_features * kernal_size, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, L):
        N, M, Fin = x.shape
        # Transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # M x Fin x N
        x0 = x0.view(M, Fin * N)  # M x Fin*N
        x = x0.unsqueeze(0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = x_.unsqueeze(0)  # 1 x M x Fin*N
            return torch.cat((x, x_), 0)  # K x M x Fin*N

        if self.K > 1:
            x1 = torch.spmm(L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * torch.spmm(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = x.view(self.K, M, Fin, N)  # K x M x Fin x N
        x = x.permute(3, 1, 2, 0).contiguous()  # N x M x Fin x K
        x = x.view(N * M, Fin * self.K)  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        # W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = torch.mm(x, self.weight) + self.bias  # N*M x Fout
        return self.activation(x.view(N, M, -1))  # N x M x Fout

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


class FeaStConv(MessagePassing):
    def __init__(self,
                 num_pts,
                 in_channels,
                 kernal_size,
                 out_channels,
                 activation='elu', 
                 bias=True):    
        super(FeaStConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = 8
        self.t_inv = True

        self.weight = nn.Parameter(torch.Tensor(in_channels,
                                             self.heads * out_channels))
        self.u = nn.Parameter(torch.Tensor(in_channels, self.heads))
        self.c = nn.Parameter(torch.Tensor(self.heads))
        if not self.t_inv:
            self.v = nn.Parameter(torch.Tensor(in_channels, self.heads))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal(self.weight, mean=0, std=0.1)
        normal(self.u, mean=0, std=0.1)
        normal(self.c, mean=0, std=0.1)
        normal(self.bias, mean=0, std=0.1)
        if not self.t_inv:
            normal(self.v, mean=0, std=0.1)

    def forward(self, x, t_vertex, edge_index):
        edge_index, _ = remove_self_loops(edge_index[0])
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j):
        # dim: x_i, [E, F_in];
        if self.t_inv:
            # with translation invariance
            q = torch.mm((x_i - x_j), self.u) + self.c  #[E, heads]
        else:
            q = torch.mm(x_i, self.u) + torch.mm(x_j, self.v) + self.c
        q = F.softmax(q, dim=1)  #[E, heads]

        x_j = torch.mm(x_j, self.weight).view(-1, self.heads,
                                              self.out_channels)
        return (x_j * q.view(-1, self.heads, 1)).sum(dim=1)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)