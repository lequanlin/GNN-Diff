import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.models import GIN
from torch_geometric.nn.dense.linear import Linear

class GIN2(nn.Module):
    def __init__(self, in_channels, out_channels, num_hid=16, dropout=0.1):
        super().__init__()
        self.model_name = 'GIN'
        self.model = GIN(in_channels, num_hid, 2, out_channels, dropout)


    def forward(self, x, edge_index):
        x = self.model(x, edge_index)
        return x

def GIN_2_conv(in_channels, out_channels, num_hid, dropout):
    return GIN2(in_channels, out_channels, num_hid, dropout)

# For debug
if __name__ == '__main__':
    model = GIN_2_conv(1433, 7, 16, 0.1)
    print('END')