from data.data_process import link_k_shot_split
from data.data_loader import load_pretrain_single_graph_data
from configs.pretrain_config import parse_pretrain_config

if __name__ == '__main__':
    configs = parse_pretrain_config()
    data_name = "WordNet18RR"
    configs.root = "../datasets"
    configs.kg_model = "transe"
    configs.kg_batch_size = 1000
    configs.kg_epochs = 500
    configs.k_hops = 2
    dataset = link_k_shot_split(configs, data_name, 5, 3)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for batch in loader:
    #     pass