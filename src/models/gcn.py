import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,global_add_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_add_pool(x, batch)
        x = self.lin(x)

        return x