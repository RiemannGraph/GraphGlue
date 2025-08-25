from data.data_loader import load_pretrain_single_graph_data
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T


if __name__ == '__main__':
    from utils.configs import DotDict
    configs = DotDict({})
    configs.data_name = "ogbn-arxiv"
    configs.root = "../datasets"
    configs.kg_model = "transe"
    configs.kg_batch_size = 1000
    configs.kg_epochs = 500
    configs.k_hops = 2
    data = load_pretrain_single_graph_data(configs)
    loader = NeighborLoader(data, [10, 10], batch_size=64)
    trans = T.RootedEgoNets(configs.k_hops)
    for data in loader:
        data = trans(data)
        continue