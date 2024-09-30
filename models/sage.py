import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.dense.linear import Linear


class SAGE_lr(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'SAGE'
        self.dropout = dropout
        self.num_layers = num_layers

        # Create layers dynamically
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, num_hid))
        self.norms.append(nn.LayerNorm(num_hid))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(num_hid, num_hid))
            self.norms.append(nn.LayerNorm(num_hid))

        self.convs.append(SAGEConv(num_hid, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight)
        return x

class SAGE2(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'SAGE'
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, num_hid)
        self.conv2 = SAGEConv(num_hid, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class SAGE3(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'SAGE'
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, num_hid)
        self.conv2 = SAGEConv(num_hid, num_hid)
        self.conv3 = SAGEConv(num_hid, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return x


def SAGE_2_conv(in_channels, out_channels, num_hid, dropout):
    return SAGE2(in_channels, out_channels, num_hid, dropout)

def SAGE_3_conv(in_channels, out_channels, num_hid, dropout):
    return SAGE3(in_channels, out_channels, num_hid, dropout)

def SAGE_lr_conv(in_channels, out_channels, num_layers, num_hid, dropout):
    return SAGE_lr(in_channels, out_channels, num_layers, num_hid, dropout)

# For debug
if __name__ == '__main__':
    model = SAGE_2_conv(3703, 64, 64, 0.1)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")