from data.data_loader import (
    load_few_shot_single_graph_data, load_few_shot_multi_graph_data
)
from cores.configs import parse_config

if __name__ == '__main__':
    configs = parse_config()
    data_name = "PROTEINS"
    configs.root = "../datasets"
    configs.kg_model = "transe"
    configs.kg_batch_size = 1000
    configs.kg_epochs = 500
    configs.k_hops = 2
    dataset = load_few_shot_multi_graph_data(configs, data_name, 5, 3)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for batch in loader:
    #     pass