import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import ChebConv
from torch_geometric.nn.dense.linear import Linear

class Cheb1(nn.Module):
    def __init__(self, in_channels, out_channels, K = 2):
        super().__init__()
        self.model_name = 'ChebNet'
        self.conv = ChebConv(in_channels, out_channels, K)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x,edge_index,edge_weight)
        return x

class Cheb2(nn.Module):
    def __init__(self, in_channels, out_channels, K = 2, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'ChebNet'
        self.dropout = dropout
        self.conv1 = ChebConv(in_channels, num_hid, K)
        self.conv2 = ChebConv(num_hid, out_channels, K)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def Cheb_1_conv(in_channels,out_channels, K = 2):
    return Cheb1(in_channels, out_channels, K)

def Cheb_2_conv(in_channels, out_channels, K = 2, num_hid = 16, dropout = 0.1):
    return Cheb2(in_channels, out_channels, K, num_hid, dropout)

# For debug
if __name__ == '__main__':
    model = Cheb_2_conv(3703, 32, 2, 32, 0.1)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")