import torch
from cores.models import PTGB, PooLedSubgraphGNN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from data.data_loader import EgoGraphDataset
import time


if __name__ == '__main__':
    ptgb = PTGB(2, 8, 32)
    gnn = PooLedSubgraphGNN("gcn", 2, 8, 16, bias=True, norm_str="none", act_str="relu", drop=0.1)
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3], [2, 4], [3, 2], [3, 4], [4, 2], [4, 3]]).t()
    graph = Data(x, edge_index)
    dataset = EgoGraphDataset(graph, 2, in_dim=8)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    z_tans = []
    for graph in loader:
        aug_graphs = ptgb(graph.x, graph.edge_index, graph.edge_weight, graph.batch, graph.batch_size)
        z_tan = []
        for aug_graph in aug_graphs:
            tan = gnn(aug_graph.x, aug_graph.edge_index, aug_graph.edge_weight, aug_graph.batch)
            z_tan.append(tan)
        z_tan = torch.stack(z_tan, dim=0)
        z_tans.append(z_tan)
    z_tans = torch.cat(z_tans, dim=1)
    print(z_tans.shape)