import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_geometric.nn.dense.linear import Linear
from torch.nn import LayerNorm

class SGC_lr(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid, K=1, dropout=0.1):
        super().__init__()
        self.model_name = 'SGC'
        self.dropout = dropout
        self.lin1 = Linear(in_channels, num_hid)
        self.conv = SGConv(num_hid, num_hid, K)
        self.lin2 = Linear(num_hid, out_channels)
        self.ln1 = LayerNorm(num_hid)
        self.ln2 = LayerNorm(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.gelu(self.lin1(x))
        x = self.ln1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(self.conv(x, edge_index, edge_weight))
        x = self.ln2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
class SGC2(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid = 16, K = 1, dropout = 0.1):
        super().__init__()
        self.model_name = 'SGC'
        self.dropout = dropout
        self.conv = SGConv(in_channels, num_hid, K)
        self.lin = Linear(num_hid, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, K = 1):
        super().__init__()
        self.model_name = 'SGC'
        self.conv = SGConv(in_channels, out_channels, K)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        return x

def SGC_conv(in_channels, out_channels, K):
    return SGC(in_channels, out_channels, K)

def SGC_2_conv(in_channels, out_channels, num_hid, K, dropout = 0.1):
    return SGC2(in_channels, out_channels, num_hid, K, dropout)

def SGC_lr_conv(in_channels, out_channels, num_hid, K, dropout):
    return SGC_lr(in_channels, out_channels, num_hid, K, dropout)

# For debug
if __name__ == '__main__':
    model = SGC_lr_conv(14, 81, 256, 10, 0)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")