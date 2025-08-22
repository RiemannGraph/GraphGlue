import torch_geometric.transforms as T


class SingleGraphDataset:
    """单图数据集包装器"""
    def __init__(self, graph, k_hops=2):
        self.graph = graph
        self.ego_net = T.RootedEgoNets(k_hops)

class MultiGraphDataset:
    """多图数据集包装器"""
    def __init__(self, graphs):
        self.graphs = graphs