import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge
from torch_geometric.nn.dense.linear import Linear


class GCN_lr(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, num_layers=2, dropout=0.1):
        super().__init__()
        self.model_name = 'GCN'
        self.dropout = dropout
        self.num_layers = num_layers

        # Create layers
        self.convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.convs.append(Linear(in_channels, num_hid))
        self.ln.append(nn.LayerNorm(num_hid))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(num_hid, num_hid))
            self.ln.append(nn.LayerNorm(num_hid))

        self.convs.append(Linear(num_hid, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            if i > 0:
                x_in = x
                x = self.convs[i](x, edge_index, edge_weight)
                x = self.ln[i](x)
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training) + x_in
            else:
                x = self.convs[i](x)
                x = self.ln[i](x)
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # Last layer without residual connection
        x = self.convs[-1](x)
        return x

class GCN1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model_name = 'GCN'
        self.conv = GCNConv(in_channels,out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x,edge_index,edge_weight)
        return x

class GCN2(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'GCN'
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid)
        self.conv2 = GCNConv(num_hid, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GCN3(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'GCN'
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid)
        self.conv2 = GCNConv(num_hid, num_hid)
        self.conv3 = GCNConv(num_hid, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return x


def GCN_conv(in_channels, out_channels, num_hid, num_layers, dropout):
    return GCN(in_channels, out_channels, num_hid, num_layers, dropout)

def GCN_1_conv(in_channels,out_channels):
    return GCN1(in_channels, out_channels)

def GCN_2_conv(in_channels, out_channels, num_hid, dropout):
    return GCN2(in_channels, out_channels, num_hid, dropout)

def GCN_3_conv(in_channels, out_channels, num_hid, dropout):
    return GCN3(in_channels, out_channels, num_hid, dropout)

def GCN_lr_conv(in_channels, out_channels, num_hid, num_layers, dropout):
    return GCN_lr(in_channels, out_channels, num_hid, num_layers, dropout)

# For debug
if __name__ == '__main__':
    model = GCN_2_conv(14,81,192,0.5)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")