from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv, DenseGCNConv
from layers import SGCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)

        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, data.batch)
        # x = global_mean_pool(x, data.batch)
        # x = global_max_pool(x, data.batch)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return F.log_softmax(x)


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()

        self.conv1 = GATConv(num_node_features, 32, heads=1)
        self.conv2 = GATConv(32, 64, heads=1)
        self.conv3 = GATConv(64, 128, heads=1)

        self.linear1 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, data.batch)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return F.log_softmax(x)


class SGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SGCN, self).__init__()
        self.conv1 = SGCNConv(coors=2, out_channels_1=64, out_features=16)
        self.conv2 = SGCNConv(coors=2, out_channels_1=64, out_features=16)
        self.conv3 = SGCNConv(coors=2, out_channels_1=64, out_features=16)

        self.linear = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        x = x[:, 1].unsqueeze(1)

        x = self.conv1(x, edge_index, pos)
        # x = F.relu(x)
        x = self.conv2(x, edge_index, pos)
        # x = F.relu(x)
        x = self.conv3(x, edge_index, pos)

        x = global_add_pool(x, data.batch)
        # x = global_mean_pool(x, data.batch)
        # x = global_max_pool(x, data.batch)

        x = self.linear(x)

        return F.log_softmax(x)
