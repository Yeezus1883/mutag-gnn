import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout):
        super().__init__()

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = global_add_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x