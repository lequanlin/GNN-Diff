import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv
from torch_geometric.nn.dense.linear import Linear
from torch.nn import LayerNorm

class MixHop_lr(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1, powers=[6, 8, 10]):
        super().__init__()
        self.model_name = 'MixHop'
        self.dropout = dropout
        self.lin1 = Linear(in_channels, num_hid)
        self.conv = MixHopConv(num_hid, num_hid, powers)
        self.lin2 = Linear(num_hid * len(powers), out_channels)  # num_hid * len(powers) instead of num_hid * 3
        self.ln1 = LayerNorm(num_hid)
        self.ln2 = LayerNorm(num_hid * len(powers))  # Adjusting for the concatenated output size

    def forward(self, x, edge_index, edge_weight=None):
        x = F.gelu(self.lin1(x))
        x = self.ln1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(self.conv(x, edge_index, edge_weight))
        x = self.ln2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

class MixHop(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1, powers = [0,1,2]):
        super().__init__()
        self.model_name = 'MixHop'
        self.dropout = dropout
        self.conv = MixHopConv(in_channels, num_hid, powers)
        self.lin = Linear(num_hid*3, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x



def MixHop_conv(in_channels, out_channels, num_hid, dropout):
    return MixHop(in_channels, out_channels, num_hid, dropout)

def MixHop_lr_conv(in_channels, out_channels, num_hid, dropout):
    return MixHop_lr(in_channels, out_channels, num_hid, dropout)

# For debug
if __name__ == '__main__':
    model = MixHop_lr_conv(14, 81, 192, 0)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")