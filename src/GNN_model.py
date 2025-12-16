import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

class CrowdGNN(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data, edge_weight = None):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        x = self.conv1(x, edge_index, edge_weight = edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight = edge_weight)
        return x.view(-1)  # flatten to shape [num_nodes_total_in_batch]
