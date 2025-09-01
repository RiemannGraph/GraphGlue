from data.data_process import link_k_shot_split
from data.data_loader import load_few_shot_link_graph_data
from configs.pretrain_config import parse_pretrain_config
from torch_geometric.sampler import BaseSampler

if __name__ == '__main__':
    configs = parse_pretrain_config()
    data_name = "WordNet18RR"
    configs.root = "../datasets"
    configs.kg_model = "transe"
    configs.kg_batch_size = 1000
    configs.kg_epochs = 10
    configs.k_hops = 2
    configs.kg_dim = 128
    data = load_few_shot_link_graph_data(configs, data_name, 5, 3, 0.1, 0.2)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for batch in loader:
    #     pass