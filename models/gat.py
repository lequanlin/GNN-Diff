import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge
from torch_geometric.nn.dense.linear import Linear

class GAT1(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super().__init__()
        self.model_name = 'GAT'
        self.conv = GATConv(in_channels,out_channels,heads)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x,edge_index,edge_weight)
        return x

class GAT2(nn.Module):
    def __init__(self, in_channels, out_channels, heads, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'GAT'
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, num_hid, heads)
        self.conv2 = GATConv(num_hid * heads, out_channels, heads)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def GAT_1_conv(in_channels,out_channels,heads = 4):
    return GAT1(in_channels, out_channels,heads)

def GAT_2_conv(in_channels, out_channels, heads = 4, num_hid = 16, dropout = 0.1):
    return GAT2(in_channels, out_channels, heads, num_hid, dropout)

# For debug
if __name__ == '__main__':
    model = GAT_2_conv(1703,5,8,32)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")