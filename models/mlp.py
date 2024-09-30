import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear


class MLP_lr(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, num_layers=2, dropout=0.1):
        super().__init__()
        self.model_name = 'MLP'
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(Linear(in_channels, num_hid))
        self.norms.append(nn.LayerNorm(num_hid))

        for _ in range(num_layers - 2):
            self.layers.append(Linear(num_hid, num_hid))
            self.norms.append(nn.LayerNorm(num_hid))

        self.layers.append(Linear(num_hid, out_channels))

    def forward(self, x, edge_index=None, edge_weight=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.gelu(layer(x))
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layers[-1](x)
        return x

class MLP2(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'MLP'
        self.dropout = dropout
        self.lin1 = Linear(in_channels, num_hid)
        self.lin2 = Linear(num_hid, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

def MLP_prop(in_channels, out_channels, num_hid, dropout):
    return MLP2(in_channels, out_channels, num_hid, dropout)

def MLP_lr_prop(in_channels, out_channels, num_hid, num_layers, dropout):
    return MLP_lr(in_channels, out_channels, num_hid, num_layers, dropout)

# For debug
if __name__ == '__main__':
    model = MLP_lr_prop(14,21,192,10,0)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")