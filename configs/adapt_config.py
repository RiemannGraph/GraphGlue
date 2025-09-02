import argparse
import sys
from dataclasses import dataclass
from typing import List
from configs.base_config import ModelConfig, load_config_from_yaml, save_config_to_yaml, add_model_config


@dataclass
class AdaptionConfig(ModelConfig):
    # Data
    data_name: str = "PubMed"
    pretrained_checkpoint: str = "checkpoints/pretrain/pretrain_final_model.pth"
    num_neighbors: List[int] = None
    root: str = "./datasets"

    k_hops: int = 2
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


    num_workers: int = 2

    # Task
    task_type: str = "node_cls"
    task_types: List[str] = None
    k_shot: int = 5
    num_trials: int = 10
    num_val: float = 0.5
    num_test: float = 0.5

    # Training
    align_coef: float = 0.1
    batch_size: int = 128
    lr_task: float = 1e-3
    task_weight_decay: float = 1e-5
    task_epochs: int = 500
    max_grad_norm: float = 1.0
    eval_interval: int = 10
    resume_checkpoint: bool = False
    patience: int = 20

    # Path
    checkpoint_dir: str = None
    log_path: str = None


def get_pretrain_parser():
    parser = argparse.ArgumentParser(description="Graph Downstream Adaption Configuration")
    parser.add_argument("--data_name", type=str, default="PubMed",
                        help="Name of the dataset. [PubMed, Computers, FacebookPagePage, WordNet18RR, PROTEINS, HIV] ")
    parser.add_argument("--pretrained_checkpoint", type=str, default="checkpoints/pretrain/pretrain_final_model.pth",
                        help="file path of pretrained model checkpoint.")
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[10, 10],
                        help="List of number of neighbors for each hop (e.g., --num_neighbors [10 10]).")
    parser.add_argument("--root", type=str, default="./datasets", help="Root directory for datasets.")
    # Node2Vec parameters (for KGs without node features)
    parser.add_argument('--nv_dim', type=int, default=128,
                        help='Node2Vec embedding dimension (default: 128)')
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
    parser.add_argument("--k_hops", type=int, default=2,
                        help="Number of hops for subgraph extraction.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')

    # Task
    parser.add_argument("--task_type", type=str, default="node_cls", choices=["node_cls", "graph_cls", "link_pred"],
                        help="Type of downstream task.")
    parser.add_argument("--k_shot", type=int, default=5,
                        help="Number of shots in few-shot learning.")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of independent trials.")
    parser.add_argument("--num_val", type=float, default=0.1,
                        help="Proportion of validation set.")
    parser.add_argument("--num_test", type=float, default=0.2,
                        help="Proportion of test set.")

    # Training
    parser.add_argument("--align_coef", type=float, default=0.1,
                        help="Coefficient for alignment loss.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for task training.")
    parser.add_argument("--lr_task", type=float, default=1e-3,
                        help="Learning rate for task model.")
    parser.add_argument("--task_weight_decay", type=float, default=1e-5,
                        help="Weight decay for task optimizer.")
    parser.add_argument("--task_epochs", type=int, default=500,
                        help="Number of epochs for task training.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping.")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Log every N epochs.")
    parser.add_argument("--resume_checkpoint", action="store_true",
                        help="Whether to resume from checkpoint.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience for early stopping.")

    # Config IO
    parser.add_argument('--config_save_path', type=str, default=None,
                        help='Path to save current config as YAML (optional)')
    parser.add_argument('--config_load_path', type=str, default=None,
                        help='Path to load config from YAML (optional, will override cmd args)')

    add_model_config(parser)
    return parser


def parse_adaption_config() -> AdaptionConfig:
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

    config = AdaptionConfig(
        pretrain_single_graph_data=args.pretrain_single_graph_data,
        pretrain_multi_graph_data=args.pretrain_multi_graph_data,
        data_name=args.data_name,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_neighbors=args.num_neighbors,
        root=args.root,

        nv_dim=args.nv_dim,
        nv_batch_size=args.nv_batch_size,
        nv_walk_length=args.nv_walk_length,
        nv_context_size=args.nv_context_size,
        nv_lr=args.nv_lr,
        nv_walks_per_node=args.nv_walks_per_node,
        nv_p=args.nv_p,
        nv_q=args.nv_q,
        nv_num_epochs=args.nv_num_epochs,

        task_type=args.task_type,
        k_shot=args.k_shot,
        num_trials=args.num_trials,
        num_val=args.num_val,
        num_test=args.num_test,
        align_coef=args.align_coef,
        batch_size=args.batch_size,
        lr_task=args.lr_task,
        task_weight_decay=args.task_weight_decay,
        task_epochs=args.task_epochs,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        resume_checkpoint=args.resume_checkpoint,
        patience=args.patience
    )

    if args.config_save_path:
        save_config_to_yaml(config, args.config_save_path)

    # Paths
    config.log_path = f"logs/{config.task_type}/{config.k_shot}-shot/{config.data_name}.log"
    config.checkpoint_dir = f"checkpoints/{config.task_type}/{config.k_shot}-shot/{config.data_name}/"

    return config