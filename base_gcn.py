import torch
from torch_geometric.nn.conv import GCNConv
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, number_nodes: int,):
        super(GCN, self).__init__()

        self.bn = nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, h_dim)
        self.conv2 = GCNConv(h_dim, out_dim)

        self.general_mlp = nn.Sequential(
            nn.Linear(3 * out_dim, 2 * out_dim),
            nn.ReLU(),
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2)
        )
        self.general_mlp2 = nn.Sequential(
            nn.Linear(3 * out_dim, 4 * out_dim),
            nn.ReLU(),
            nn.Linear(4 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim // 2)
        )
        self.edge_lin1 = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim)
        )

        self.classifier = nn.Linear(in_features=out_dim, out_features=7)
        self.number_nodes = number_nodes

    def forward(self, x, edge_index):
        x = self.bn(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def pred(self, x, idx):

        xs = [x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]]
        xs = torch.cat(xs, dim=1)

        xs = self.general_mlp(xs)
        return xs

    def pooling2(self, x, idx):

        xs = [F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 1]])),
              F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 2]])),
              F.relu(self.edge_lin1(x[idx[:, 1]] - x[idx[:, 2]])),]
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp2(xs)
        return xs


class GCN_binary(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, number_nodes: int, args):
        super(GCN_binary, self).__init__()

        self.bn = nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, h_dim)
        self.conv2 = GCNConv(h_dim, out_dim)
        if args.abla_edge:
            basic_out = out_dim * 3
        else:
            basic_out = out_dim // 2 * 3
        if not args.abla_basic:
            self.general_mlp = nn.Sequential(
                nn.Linear(3 * out_dim, 2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, basic_out)
            )
        else:
            self.general_mlp = nn.Identity()

        if not args.abla_edge:
            self.general_mlp2 = nn.Sequential(
                nn.Linear(3 * out_dim, 4 * out_dim),
                nn.ReLU(),
                nn.Linear(4 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim // 2 * 3)
            )
            self.edge_lin1 = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim)
            )

        self.classifier = nn.Linear(out_dim * 3, 2)
        self.number_nodes = number_nodes

    def forward(self, x, edge_index):
        x = self.bn(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def pred(self, x, idx):

        xs = [x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]]
        xs = torch.cat(xs, dim=1)

        xs = self.general_mlp(xs)
        return xs

    def pooling2(self, x, idx):

        xs = [F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 1]])),
              F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 2]])),
              F.relu(self.edge_lin1(x[idx[:, 1]] - x[idx[:, 2]])),]
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp2(xs)
        return xs

