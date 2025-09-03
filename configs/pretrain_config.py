import sys
import argparse
from dataclasses import dataclass
from typing import List
from configs.base_config import ModelConfig, add_model_config, load_config_from_yaml, save_config_to_yaml


@dataclass
class PretrainConfig(ModelConfig):
    # Data
    num_neighbors: List[int] = None

    root: str = "./datasets"

    # For Node2Vec, data like KG that without node features
    nv_dim: int = 128
    nv_batch_size: int = 128
    nv_walk_length: int = 20
    nv_context_size: int = 10
    nv_lr: float = 0.01
    nv_walks_per_node: int = 10
    nv_p: float = 1.0
    nv_q: float = 1.0
    nv_num_epochs: int = 100

    k_hops: int = 2
    batch_size: int = 128
    num_workers: int = 0

    # Training
    pretrain_epochs: int = 100
    lr_pretrain: float = 1e-3
    pretrain_weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 10
    inter_loss_interval: int = 1
    resume_checkpoint: bool = False
    resume_temp_checkpoint: bool = False
    # Loss & Graph
    knn: int = 15

    # Path
    checkpoint_dir: str = "checkpoints/pretrain/"
    log_path: str = "logs/pretrain/pretrain.log"


def get_pretrain_parser():
    parser = argparse.ArgumentParser(description="Graph Pretraining Configuration")

    # Data
    parser.add_argument('--root', type=str, default="./datasets")

    # Node2Vec parameters (for KGs without node features)
    parser.add_argument('--nv_dim', type=int, default=128,
                        help='dimension of node2vec embedding')
    parser.add_argument('--nv_batch_size', type=int, default=128,
                        help='Batch size for Node2Vec training (default: 128)')
    parser.add_argument('--nv_walk_length', type=int, default=20,
                        help='Length of random walks in Node2Vec (default: 20)')
    parser.add_argument('--nv_context_size', type=int, default=10,
                        help='Context size for context-target prediction (default: 10)')
    parser.add_argument('--nv_lr', type=float, default=0.01,
                        help='Learning rate for Node2Vec optimizer (default: 0.01)')
    parser.add_argument('--nv_walks_per_node', type=int, default=10,
                        help='Number of random walks per node (default: 10)')
    parser.add_argument('--nv_p', type=float, default=1.0,
                        help='Return parameter in Node2Vec (default: 1.0)')
    parser.add_argument('--nv_q', type=float, default=1.0,
                        help='In-out parameter in Node2Vec (default: 1.0)')
    parser.add_argument('--nv_num_epochs', type=int, default=100,
                        help='Number of epochs to train Node2Vec (default: 100)')

    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 5],
                        help='neighbor number of each hop')
    parser.add_argument('--k_hops', type=int, default=2,
                        help='subgraph sample hops <= len(num_neighbors)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')

    # Training
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help='Total pretrain epochs')
    parser.add_argument('--lr_pretrain', type=float, default=3e-5,
                        help='Learning rate for pretraining')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--inter_loss_interval', type=int, default=1,
                        help="Compute inter_loss every N epochs")
    parser.add_argument('--resume_checkpoint', action='store_true',
                        help='Whether to resume from latest checkpoint')
    parser.add_argument('--resume_temp_checkpoint', action='store_true',
                        help='Whether to resume from temp checkpoint')

    # Loss & Graph
    parser.add_argument('--knn', type=int, default=30,
                        help='KNN graph connections for inter-graph loss')

    # Config IO
    parser.add_argument('--config_save_path', type=str, default="./scripts/pretrain/pretrain.yaml",
                        help='Path to save current config as YAML (optional)' \
                             'default path format is ./scripts/pretrain/pretrain.yaml')
    parser.add_argument('--config_load_path', type=str, default=None,
                        help='Path to load config from YAML (optional, will override cmd args)')

    add_model_config(parser)
    return parser


def parse_pretrain_config() -> PretrainConfig:
    parser = get_pretrain_parser()
    args = parser.parse_args()

    # If using YAML file
    if args.config_load_path:
        print(f"Loading config from YAML: {args.config_load_path}")
        yaml_config = load_config_from_yaml(args.config_load_path)

        for key, value in yaml_config.items():
            if hasattr(args, key):
                if not any(opt_str in str(sys.argv) for opt_str in [f'--{key}', f'-{key}']):
                    setattr(args, key, value)

    config = PretrainConfig(
        num_neighbors=args.num_neighbors,
        k_hops=args.k_hops,
        pretrain_single_graph_data=args.pretrain_single_graph_data,
        pretrain_multi_graph_data=args.pretrain_multi_graph_data,
        root=args.root,

        nv_batch_size=args.nv_batch_size,
        nv_walk_length=args.nv_walk_length,
        nv_context_size=args.nv_context_size,
        nv_lr=args.nv_lr,
        nv_walks_per_node=args.nv_walks_per_node,
        nv_p=args.nv_p,
        nv_q=args.nv_q,
        nv_num_epochs=args.nv_num_epochs,

        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_epochs=args.pretrain_epochs,
        lr_pretrain=args.lr_pretrain,
        pretrain_weight_decay=args.pretrain_weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_temp_checkpoint=args.resume_temp_checkpoint,
        knn=args.knn
    )

    # Path
    config.log_path = "logs/pretrain/pretrain.log"
    dir_name = ""
    for d in config.pretrain_single_graph_data:
        dir_name += f"{d}_"
    for d in config.pretrain_multi_graph_data:
        dir_name += f"{d}_"
    config.checkpoint_dir = f"checkpoints/pretrain/{dir_name[:-1]}"

    if args.config_save_path:
        save_config_to_yaml(config, args.config_save_path)
    return config