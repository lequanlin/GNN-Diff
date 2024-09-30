import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.nn.dense.linear import Linear
from torch.nn import LayerNorm

class APPNP_net_lr(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, K=2, alpha=0.5, dropout=0.1):
        super().__init__()
        self.model_name = 'APPNP'
        self.dropout = dropout
        self.lin1 = Linear(in_channels, num_hid)
        self.conv = APPNP(K, alpha, dropout)
        self.lin2 = Linear(num_hid, out_channels)
        self.ln1 = LayerNorm(num_hid)
        self.ln2 = LayerNorm(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.gelu(self.lin1(x))
        x = self.ln1(x)
        x = F.gelu(self.conv(x, edge_index, edge_weight))
        x = self.ln2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

class APPNP_net(nn.Module):
    def __init__(self, in_channels, out_channels, K = 2, alpha =0.5, dropout=0.1):
        super().__init__()
        self.model_name = 'APPNP'
        self.dropout = dropout
        self.conv = APPNP(K, alpha, dropout)
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

class APPNP_net_link(nn.Module):
    def __init__(self, in_channels, out_channels, K = 2, alpha =0.5, dropout=0.1):
        super().__init__()
        self.model_name = 'APPNP'
        self.dropout = dropout
        self.conv = APPNP(K, alpha, dropout)
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

def APPNPNet(in_channels, out_channels, K = 2, alpha =0.5, dropout=0.1):
    return APPNP_net(in_channels, out_channels, K, alpha, dropout)

def APPNPNet_LR(in_channels, out_channels, num_hid, K, alpha, dropout):
    return APPNP_net_lr(in_channels, out_channels, num_hid, K, alpha, dropout)

def APPNPNet_Link(in_channels, out_channels, K = 2, alpha =0.5, dropout=0.1):
    return APPNP_net_link(in_channels, out_channels, K, alpha, dropout)

# For debug
if __name__ == '__main__':
    model = APPNPNet_Link(1433, 64)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")