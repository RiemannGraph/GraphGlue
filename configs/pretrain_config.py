import sys
import argparse
from dataclasses import dataclass
from typing import List
from configs.base_config import ModelConfig, add_model_config, load_config_from_yaml, save_config_to_yaml


@dataclass
class PretrainConfig(ModelConfig):
    # Data
    num_neighbors: List[int] = None
    pretrain_single_graph_data: List[str] = None
    pretrain_multi_graph_data: List[str] = None

    root: str = "./datasets"
    kg_model: str = "transe"
    kg_batch_size: int = 1024
    kg_epochs: int = 500

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
    resume_checkpoint: bool = False

    # Loss & Graph
    knn: int = 15

    # Paths
    log_path: str = "logs/pretrain/pretrain.log"
    checkpoint_dir: str = "checkpoints/pretrain/"


def get_pretrain_parser():
    parser = argparse.ArgumentParser(description="Graph Pretraining Configuration")

    # Data
    parser.add_argument('--root', type=str, default="./datasets")
    parser.add_argument('--kg_model', type=str, default="transe",
                        choices=['transe', 'complex', 'distmult', 'rotate'], help="Knowledge Graph Model for KG embedding")
    parser.add_argument('--kg_batch_size', type=int, default=1024, help="batch size to training kg_model")
    parser.add_argument('--kg_epochs', type=int, default=500, help="number of training kg_model epochs")
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[5, 2],
                        help='neighbor number of each hop')
    parser.add_argument('--k_hops', type=int, default=2,
                        help='subgraph sample hops <= len(num_neighbors)')
    parser.add_argument('--pretrain_single_graph_data', type=str, nargs='+',
                        # default=["ogbn-arxiv", "AmazonProducts", "Reddit", "FB15k_237"],
                        default=["ogbn-arxiv"],
                        help='node-level pretraining datasets')
    parser.add_argument('--pretrain_multi_graph_data', type=str, nargs='+', default=["PCBA"],
                        help='graph-level pretraining datasets')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')

    # Training
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Total pretrain epochs')
    parser.add_argument('--lr_pretrain', type=float, default=1e-3,
                        help='Learning rate for pretraining')
    parser.add_argument('--pretrain_weight_decay', type=float, default=1e-5,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume_checkpoint', action='store_true',
                        help='Whether to resume from latest checkpoint')

    # Loss & Graph
    parser.add_argument('--knn', type=int, default=5,
                        help='KNN graph connections for inter-graph loss')

    # Paths
    parser.add_argument('--log_path', type=str, default="logs/pretrain/pretrain.log",
                        help='Path to log file')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/pretrain/",
                        help='Directory to save checkpoints')

    # Config IO
    parser.add_argument('--config_save_path', type=str, default=None,
                        help='Path to save current config as YAML (optional)')
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
        kg_model=args.kg_model,
        kg_batch_size=args.kg_batch_size,
        kg_epochs=args.kg_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_epochs=args.pretrain_epochs,
        lr_pretrain=args.lr_pretrain,
        pretrain_weight_decay=args.pretrain_weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        knn=args.knn,
        log_path=args.log_path,
        checkpoint_dir=args.checkpoint_dir
    )

    if args.config_save_path:
        save_config_to_yaml(config, args.config_save_path)

    return config