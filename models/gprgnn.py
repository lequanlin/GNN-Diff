import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, alpha, Init = 'Random', K = 10, Gamma=None, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha]= 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN_lr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, alpha=0.1, K=2, dropout=0.1):
        super(GPRGNN_lr, self).__init__()
        self.model_name = 'GPRGNN'
        self.lin1 = Linear(in_channels, num_hid)
        self.lin2 = Linear(num_hid, out_channels)
        self.prop1 = GPR_prop(alpha=alpha, K=K)
        self.ln1 = LayerNorm(num_hid)  # LayerNorm after lin1
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(self.lin1(x))
        x = self.ln1(x)  # Apply LayerNorm after activation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dropout == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
            return x
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)
            return x


class GPRGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, alpha = 0.1, dropout=0.1):
        super(GPRGNN, self).__init__()
        self.model_name = 'GPRGNN'
        self.lin1 = Linear(in_channels, num_hid)
        self.lin2 = Linear(num_hid, out_channels)

        self.prop1 = GPR_prop(alpha = alpha)

        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dropout == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)
            return x


def GPRGNN_conv(in_channels, out_channels, num_hid, alpha, dropout):
    return GPRGNN(in_channels, out_channels, num_hid, alpha, dropout)

def GPRGNN_lr_conv(in_channels, out_channels, num_hid, alpha, K, dropout):
    return GPRGNN_lr(in_channels, out_channels, num_hid, alpha, K, dropout)

# For debug
if __name__ == '__main__':
    model = GPRGNN_lr_conv(14, 21, 256, 0.1, 8, 0)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")