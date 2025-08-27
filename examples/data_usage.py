from data.data_loader import load_pretrain_single_graph_data, load_pretrain_multi_graph_data
from utils.configs import parse_config
from torch_geometric.loader import DataLoader


if __name__ == '__main__':
    configs = parse_config()
    data_name = "PCBA"
    configs.root = "../datasets"
    configs.kg_model = "transe"
    configs.kg_batch_size = 1000
    configs.kg_epochs = 500
    configs.k_hops = 2
    dataset = load_pretrain_multi_graph_data(configs, data_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in loader:
        pass