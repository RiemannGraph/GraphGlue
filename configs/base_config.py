from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from argparse import ArgumentParser
import yaml
import os


@dataclass
class ModelConfig:

    pretrain_single_graph_data: List[str] = None
    pretrain_multi_graph_data: List[str] = None

    """Shared model architecture configuration"""

    n_layers: int = 3
    num_samples: Optional[int] = None
    in_dim: int = 100
    hid_dim: int = 256
    att_dim: int = 512
    num_generators: int = 128
    bias: bool = True
    act_str: str = "gelu"
    drop: float = 0.1
    conv_name: str = "gcn"
    normalize: bool = True
    norm_str: str = "layer_norm"
    temperature: float = 1.0

    regular_coef_pt: float = 0.5
    regular_coef_curv: float = .5


def add_model_config(parser: ArgumentParser):
    """Add shared model architecture arguments"""
    group = parser.add_argument_group("Model Architecture")
    group.add_argument('--pretrain_single_graph_data', type=str, nargs='+',
                        # default=["ogbn-arxiv", "AmazonProducts", "Reddit", "FB15k_237"],
                        default=["ogbn-arxiv"],
                        help='node-level pretraining datasets')
    group.add_argument('--pretrain_multi_graph_data', type=str, nargs='+', default=["PCBA"],
                        help='graph-level pretraining datasets')
    group.add_argument('--n_layers', type=int, default=3, help='Number of GNN layers')
    group.add_argument('--num_samples', type=int, default=None, help='Number of adjacent edge samples')
    group.add_argument('--in_dim', type=int, default=100, help='Input feature dimension')
    group.add_argument('--hid_dim', type=int, default=256, help='Hidden dimension')
    group.add_argument('--att_dim', type=int, default=512, help='Attention dimension (if used)')
    group.add_argument('--num_generators', type=int, default=128, help='Number of generators in FM')
    group.add_argument('--conv_name', type=str, default='gcn', choices=['gcn', 'sage', 'gat'], help='GNN layer type')
    group.add_argument('--act_str', type=str, default='gelu', help='Activation function')
    group.add_argument('--drop', type=float, default=0.1, help='Dropout rate')
    group.add_argument('--normalize', action='store_true', help='Whether to normalize adjacency matrix')
    group.add_argument('--bias', action='store_false', help='Whether to add bias term')
    group.add_argument('--norm_str', type=str, default='layer_norm', choices=['layer_norm', 'batch_norm'])
    group.add_argument('--temperature', type=float, default=1.0, help='Temperature')
    group.add_argument('--regular_coef_pt', type=float, default=0.5, help='Regularization coefficient of PT')
    group.add_argument('--regular_coef_curv', type=float, default=0.5, help='Regularization coefficient of Curvature')
    return parser


def save_config_to_yaml(config: ModelConfig, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(asdict(config), f, default_flow_style=False, indent=2, sort_keys=False)
    print(f"Config saved to {filepath}")


def load_config_from_yaml(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# def add_training_config(parser: ArgumentParser):
#     """Add common training arguments"""
#     group = parser.add_argument_group("Training Parameters")
#     group.add_argument('--batch_size', type=int, default=128)
#     group.add_argument('--num_workers', type=int, default=0)
#     group.add_argument('--max_grad_norm', type=float, default=1.0)
#     group.add_argument('--log_interval', type=int, default=10)
#     return parser
#
# def add_path_config(parser: ArgumentParser):
#     """Add path-related arguments"""
#     group = parser.add_argument_group("Paths")
#     group.add_argument('--root', type=str, default='./datasets')
#     group.add_argument('--log_path', type=str, default='logs/default.log')
#     group.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
#     return parser