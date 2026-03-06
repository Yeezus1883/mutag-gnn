import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool


class GAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_dim, out_channels, heads, dropout):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_add_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)

        return x