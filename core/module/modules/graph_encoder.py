import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear


"""

The idea of Graph VAE is proposed in "Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. arXiv preprint arXiv:1611.07308."
The code here is its PyTorch implementation from https://github.com/zfjsail/gae-pytorch

"""

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.1, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class GraphVAEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, type = 'AE'):
        super(GraphVAEncoder, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc3 = GraphConvolution(input_feat_dim, hidden_dim2, dropout, act=F.relu)

        self.lin1 = nn.Linear(input_feat_dim,hidden_dim1)
        self.lin2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.lin3 = nn.Linear(hidden_dim2 * 3, hidden_dim2)

        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(1.0))

        self.act = nn.ReLU()
        self.type = type
        self.dropout = dropout

        # Create layers for LR-version
        self.convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.convs.append(Linear(input_feat_dim, hidden_dim2))
        self.ln.append(nn.LayerNorm(hidden_dim2))

        for _ in range(6):
            self.convs.append(GCNConv(hidden_dim2,hidden_dim2))
            self.ln.append(nn.LayerNorm(hidden_dim2))

    def encode_VAE(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def encode_AE(self, x, adj):

        hidden = []

        hidden_g_1 = self.gc2(self.gc1(x, adj), adj)
        hidden.append(hidden_g_1)
        hidden_g_2 = self.gc3(x, adj)
        hidden.append(hidden_g_2)

        x = F.dropout(x,self.dropout, self.training)
        hidden_l = self.act(self.lin1(x))
        hidden_l = F.dropout(hidden_l,self.dropout, self.training)
        hidden.append(hidden_l)

        return self.lin3(torch.cat(hidden, dim=1))

    def encode_AE_LR(self, x, edge_index, edge_weight = None):
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
        return x



    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        if self.type == 'AE':
            z = self.encode_AE(x, adj)
            return z, z, None #To match VAE
        elif self.type == 'VAE':
            mu, logvar = self.encode_VAE(x, adj)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar


class GraphVAEncoder_Link(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim2, dropout):
        super(GraphVAEncoder_Link, self).__init__()

        self.dropout = dropout

        self.conv1 = GCNConv(input_feat_dim, hidden_dim2)
        self.conv2 = GCNConv(hidden_dim2, hidden_dim2)

    def encode_AE_Link(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def forward(self, x, edge_index, edge_weight = None):
        z = self.encode_AE_Link(x, edge_index, edge_weight)
        return z

class GraphVAEncoder_LR(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim2, dropout):
        super(GraphVAEncoder_LR, self).__init__()

        self.dropout = dropout

        # Create layers for LR-version
        self.convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.convs.append(Linear(input_feat_dim, hidden_dim2))
        self.ln.append(nn.LayerNorm(hidden_dim2))

        for _ in range(6):
            self.convs.append(GCNConv(hidden_dim2, hidden_dim2))
            self.ln.append(nn.LayerNorm(hidden_dim2))

    def encode_AE_LR(self, x, edge_index, edge_weight=None):
        for i in range(7):
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
        return x

    def forward(self, x, edge_index, edge_weight = None):
        z = self.encode_AE_LR(x, edge_index, edge_weight)
        return z

class GraphVADecoder_node(nn.Module):

    """Decoder for node classification."""

    def __init__(self, hidden_dim2, output_dim, dropout):
        super(GraphVADecoder_node, self).__init__()
        self.gc = GraphConvolution(hidden_dim2, output_dim, dropout = 0, act=lambda x: x)
        self.lin = nn.Linear(hidden_dim2, output_dim)
        self.dropout = dropout
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, adj):
        z = F.dropout(z, self.dropout, training=self.training)
        c = self.lin(z)
        return F.log_softmax(c, dim=1)

class GraphVADecoder_Link(nn.Module):

    """Decoder for link prediction."""

    def __init__(self, hidden_dim, dropout):
        super(GraphVADecoder_Link, self).__init__()
        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        outputs = self.lin(z)
        return outputs

class GraphVADecoder_LR(nn.Module):

    """Decoder for node classification on long-range graphs."""

    def __init__(self, hidden_dim, output_dim, dropout):
        super(GraphVADecoder_LR, self).__init__()
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        outputs = self.lin(z)
        return outputs

class GraphVAEncoder2Decoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim = 7, dropout = 0.5, input_noise_factor=0.001, latent_noise_factor=0.1, task = 'node_classification'):
        super(GraphVAEncoder2Decoder, self).__init__()

        self.input_feat_dim = input_feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout

        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor

        self.task = task


        if self.task in ["node_classification", "node_classification_large"]:
            self.encoder = GraphVAEncoder(input_feat_dim, hidden_dim1, hidden_dim2, dropout)
            self.decoder = GraphVADecoder_node(hidden_dim2, output_dim,dropout)
        elif self.task in ['node_classification_lr']:
            self.encoder = GraphVAEncoder_LR(input_feat_dim, hidden_dim2, dropout)
            self.decoder = GraphVADecoder_LR(hidden_dim2, output_dim, dropout)
        elif self.task in ['link_prediction']:
            self.encoder = GraphVAEncoder_Link(input_feat_dim, hidden_dim2, dropout)
            self.decoder = GraphVADecoder_Link(hidden_dim2, dropout)

    def encode(self, x, adj, **kwargs):
        return self.encoder(x, adj, **kwargs)

    def decode(self, x, adj, **kwargs):
        return self.decoder(x, adj, **kwargs)

    def encode_lr(self, x, edge_index, edge_weight = None, **kwargs):
        return self.encoder(x, edge_index, edge_weight)

    def decode_lr(self, x, **kwargs):
        return self.decoder(x)

    def encode_lp(self, x, edge_index, edge_weight = None, **kwargs):
        return self.encoder(x, edge_index, edge_weight)

    def decode_lp(self, x, **kwargs):
        return self.decoder(x)

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def forward(self, x, adj = None, edge_index = None, edge_weight = None, **kwargs):

        x = self.add_noise(x, self.input_noise_factor)
        if self.task in ['node_classification', 'node_classification_large', ]:
            x, mu, logvar = self.encode(x, adj)
        elif self.task in ['node_classification_lr']:
            x = self.encode_lr(x, edge_index, edge_weight)
            mu = None
            logvar = None
        elif self.task in ['link_prediction']:
            x = self.encode_lp(x, edge_index, edge_weight)
            mu = None
            logvar = None

        x = self.add_noise(x, self.latent_noise_factor)

        if self.task in ['node_classification', 'node_classification_large', ]:
            x = self.decode(x, adj)
        elif self.task in ['node_classification_lr']:
            x = self.decode_lr(x)
        elif self.task in ['link_prediction']:
            x = self.decode_lp(x)

        return x, mu, logvar

class GraphVAE_node(GraphVAEncoder2Decoder):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim, input_noise_factor, latent_noise_factor, task):
        dropout = 0.1
        super(GraphVAE_node, self).__init__(input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout, input_noise_factor, latent_noise_factor, task)

class GraphVAE_link(GraphVAEncoder2Decoder):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim, input_noise_factor, latent_noise_factor, task):
        dropout = 0.1
        output_dim = None
        super(GraphVAE_link, self).__init__(input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout, input_noise_factor, latent_noise_factor, task)

# For debug
if __name__ == '__main__':
    model = GraphVAE_node(1433, 64, 64, 7, 0.1, 0.1, task='node_classification_lr')
