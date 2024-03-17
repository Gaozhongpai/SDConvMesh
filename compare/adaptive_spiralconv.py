import torch
import torch.nn as nn

class Dynamic_spiral_pool(nn.Module): 
    def __init__(self, in_channels, out_channels, dynamic_indices=None, dim=1, store_values=True,**kwargs):
        super(Dynamic_spiral_pool, self).__init__()
        self.store_values = store_values
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.indices = dynamic_indices
        self.max_seq = dynamic_indices.size(1)        
        self.ro    = nn.Linear(in_channels , 1) 

        if in_channels == 3:
            self.norm  = nn.GroupNorm(1, in_channels) 
        else:
            self.norm  = nn.GroupNorm(4, in_channels) 

        self.reset_parameters
    
    def reset_parameters(self):     
        torch.nn.init.constant_(self.ro.weight, 0)
        torch.nn.init.constant_(self.ro.bias, 0)
    
    def dynamic_weighted_pool(self, x): 
        '''
        input: bs, n_nodes, max_seq, ch.
        '''                     
        b, n, k, c = x.size()
        assert k == self.max_seq

        ## get s
        s = torch.abs(self.ro(x.mean(2))).view(b, n, 1)  #: mean(b,n,k,c) -> ll(b,n,c) -> b,n,1  
        s = torch.clamp(s*self.max_seq, max=self.max_seq-1)
        pool= x.cumsum(2)

        _t = s.ceil().long().detach() 
        _b = s.floor().long().detach()              
        frac = s - _b
    
        it = _t.view(b,n,1,1).repeat(1,1,1,c)
        ib = _b.view(b,n,1,1).repeat(1,1,1,c)

        xt = torch.gather(pool,2, it)
        xb = torch.gather(pool,2, ib)
        
        pool_x = xb + frac.unsqueeze(-1) * (xt - xb)
        
        y = pool_x.view(b, n, -1)
        y = self.norm(y.permute(0,2,1)).permute(0,2,1)
        
        return y, s 

    def forward(self, x):
        assert x.dim() == 3   
        bs, n_nodes, _ = x.size() 
        
        x = torch.index_select(x, self.dim, self.indices.view(-1))
        x = x.view(bs, n_nodes, self.max_seq, -1)            
        x, s = self.dynamic_weighted_pool(x)         

        return x

    def __repr__(self):
        return '{}({}, {}, max_s = {})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.max_seq)

class Group_Linear(nn.Module):
    def __init__(self,in_channels, out_channels, g=2):
        super(Group_Linear, self).__init__()
        self.in_ch  = in_channels //g
        self.out_ch = out_channels//g
        self.g=g
        self.weight = nn.Parameter(torch.Tensor(self.in_ch, self.out_ch))
        self.bias =   nn.Parameter(torch.Tensor(self.out_ch))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.constant_(self.bias, 0)        
    
    def forward(self, x):
        b, n, _ = x.size()
        x = x.view(b,n,self.g,-1)
        x = torch.einsum('bngc, cd -> bngd', x, self.weight) + self.bias
        x = x.view(b,n,-1)
        return x
    
    def __repr__(self):
        return '{}({}, {}, g={})'.format(self.__class__.__name__,
                                                  self.in_ch,
                                                  self.out_ch,
                                                  self.g)

class Gated_spiral_dw(nn.Module):
    def __init__(self,in_channels, indices, dim):
        super(Gated_spiral_dw, self).__init__()
        self.dim = dim
        self.indices = indices
        self.seq_length = indices.size(1)
        self.n_nodes = indices.size(0)
        self.ch = in_channels

        self.gate   = nn.Linear(self.ch,  self.ch, bias=True)      
        self.weight = nn.Parameter(torch.Tensor(self.n_nodes, self.seq_length))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        bs, n_nodes, _ = x.size()
        _gate = self.gate(x)

        _x = torch.index_select(x, self.dim, self.indices.view(-1)) # bs x n_nodes x seq_length * ch
        _x = _x.reshape(bs, self.n_nodes, self.seq_length, self.ch)
        x = torch.einsum('bvsf, vs -> bvf', _x, self.weight) 
        return x*_gate

class SpiralConv(nn.Module): 
    def __init__(self, in_channels, out_channels, indices=None, dim=1, **kwargs):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

class Adaptive_SpiralConv_simple(nn.Module): 
    def __init__(self, in_channels, out_channels, indices=None, dynamic_indices=None,dim=1, **kwargs):
        super(Adaptive_SpiralConv_simple, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.dynamic_dw = Gated_spiral_dw(self.in_channels, self.indices,self.dim )
        self.SpiralConv = nn.Linear(in_channels * self.seq_length, out_channels)
        self.dynamic_pool   = Dynamic_spiral_pool(self.out_channels, self.out_channels, dynamic_indices=dynamic_indices)
        
        self.alpha = nn.Parameter(torch.ones([1])*0.1)
        self.beta = nn.Parameter(torch.ones([1])*1)
       
    def forward(self, x):
        assert x.dim() == 3
        n_nodes, _ = self.indices.size()
        bs = x.size(0)

        x = self.beta.view(1,1,-1)*self.dynamic_dw(x)+ x    
        
        x = torch.index_select(x, self.dim, self.indices.view(-1))
        x = x.view(bs, n_nodes, -1)
        x = self.SpiralConv(x)

        x = self.alpha* self.dynamic_pool(x) + x   

        return x
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.SpiralConv.weight)
        torch.nn.init.constant_(self.SpiralConv.bias, 0)
    
    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

class Adaptive_SpiralConv_dw(nn.Module): 
    def __init__(self, in_channels, out_channels, indices=None, dynamic_indices=None,dim=1, **kwargs):
        super(Adaptive_SpiralConv_dw, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.dynamic_dw = Gated_spiral_dw(self.in_channels, self.indices,self.dim )
        self.SpiralConv = nn.Linear(in_channels * self.seq_length, out_channels)
        
        self.beta = nn.Parameter(torch.ones([1])*1)
       
    def forward(self, x):
        assert x.dim() == 3
        n_nodes, _ = self.indices.size()
        bs = x.size(0)

        x = self.beta.view(1,1,-1)*self.dynamic_dw(x)+ x    
        
        x = torch.index_select(x, self.dim, self.indices.view(-1))
        x = x.view(bs, n_nodes, -1)
        x = self.SpiralConv(x)

        return x
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.SpiralConv.weight)
        torch.nn.init.constant_(self.SpiralConv.bias, 0)
    
    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

class Adaptive_SpiralConv(nn.Module): 
    def __init__(self, in_channels, out_channels, indices=None, dynamic_indices=None,dim=1, **kwargs):
        super(Adaptive_SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.SpiralConv = nn.Linear(in_channels * self.seq_length, out_channels)
        
        self.dynamic_pool = Dynamic_spiral_pool(in_channels, in_channels, dynamic_indices=dynamic_indices)
        self.u_cr = Group_Linear(in_channels, in_channels, g=4)  
        self.dynamic_dw = Gated_spiral_dw(self.in_channels, self.indices,self.dim )
        self.u_dr = Group_Linear(in_channels, in_channels, g=4)  
        
        self.act = nn.ELU()
        
        self.alpha = nn.Parameter(torch.ones([1])*0.1)
        self.beta = nn.Parameter(torch.ones([1])*1)
        self.gamma = nn.Parameter(torch.ones([in_channels])*0.1)
        self.delta = nn.Parameter(torch.ones([in_channels])*0.1)
        
    def forward(self, x, t_vertex, spiral_adj):
        assert x.dim() == 3
        x = x[:, :-1, :]
        n_nodes, _ = self.indices.size()
        bs = x.size(0)
        x = self.gamma.view(1,1,-1)*self.u_cr(x)  + x
        x = self.beta.view(1,1,-1)*self.dynamic_dw(x)+ x
        x = self.alpha* self.dynamic_pool(x) + x       
        x = self.delta.view(1,1,-1)*self.u_dr(x)+ x  
        
        x = torch.index_select(x, self.dim, self.indices.view(-1))
        x = x.view(bs, n_nodes, -1)
        x = self.SpiralConv(x)
        
        # add activation here
        x = self.act(x)
        
        x = torch.cat([x, -1 * torch.ones([x.shape[0], 1, x.shape[-1]]).to(x)], dim=1)
        return x
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.SpiralConv.weight)
        torch.nn.init.constant_(self.SpiralConv.bias, 0)
    
    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)