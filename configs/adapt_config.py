import argparse
import sys
from dataclasses import dataclass
from typing import List
from configs.base_config import ModelConfig, load_config_from_yaml, save_config_to_yaml


@dataclass
class AdaptionConfig(ModelConfig):
    # Data
    data_name: str = None
    pretrained_checkpoint: str = None
    num_neighbors: List[int] = None
    root: str = "./datasets"
    kg_model: str = "transe"
    kg_batch_size: int = 1024
    kg_epochs: int = 500
    k_hops: int = 2

    # Task
    task_type: str = "node_cls"
    k_shot: int = 5
    num_trails: int = 10
    num_val: float = 0.1
    num_test: float = 0.2

    # Training
    align_coef: float = 1.0
    batch_size: int = 128
    lr_task: float = 1e-3
    task_weight_decay: float = 1e-5
    task_epochs: int = 500
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 10
    resume_checkpoint: bool = False
    patience: int = 20

    # Paths
    log_path: str = f"logs/{task_type}/{k_shot}-shot/{data_name}.log"
    checkpoint_dir: str = f"checkpoints/{task_type}/{k_shot}-shot/{data_name}/"


def get_pretrain_parser():
    parser = argparse.ArgumentParser(description="Graph Downstream Adaption Configuration")
    parser.add_argument("--data_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Path to pretrained model checkpoint.")
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=None,
                        help="List of number of neighbors for each hop (e.g., --num_neighbors 10 10).")
    parser.add_argument("--root", type=str, default="./datasets", help="Root directory for datasets.")
    parser.add_argument("--kg_model", type=str, default="transe", choices=["transe", "rotate", "distmult", "complEx"],
                        help="Knowledge graph embedding model.")
    parser.add_argument("--kg_batch_size", type=int, default=1024, help="Batch size for KG training.")
    parser.add_argument("--kg_epochs", type=int, default=500, help="Number of epochs for KG training.")
    parser.add_argument("--k_hops", type=int, default=2, help="Number of hops for subgraph extraction.")

    # Task
    parser.add_argument("--task_type", type=str, default="node_cls", choices=["node_cls", "link_pred"],
                        help="Type of downstream task.")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of shots in few-shot learning.")
    parser.add_argument("--num_trails", type=int, default=10, help="Number of independent trials.")
    parser.add_argument("--num_val", type=float, default=0.1, help="Proportion of validation set.")
    parser.add_argument("--num_test", type=float, default=0.2, help="Proportion of test set.")

    # Training
    parser.add_argument("--align_coef", type=float, default=1.0, help="Coefficient for alignment loss.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for task training.")
    parser.add_argument("--lr_task", type=float, default=1e-3, help="Learning rate for task model.")
    parser.add_argument("--task_weight_decay", type=float, default=1e-5, help="Weight decay for task optimizer.")
    parser.add_argument("--task_epochs", type=int, default=500, help="Number of epochs for task training.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N epochs.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume_checkpoint", action="store_true", help="Whether to resume from checkpoint.")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping.")

    # Config IO
    parser.add_argument('--config_save_path', type=str, default=None,
                        help='Path to save current config as YAML (optional)')
    parser.add_argument('--config_load_path', type=str, default=None,
                        help='Path to load config from YAML (optional, will override cmd args)')
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
        data_name=args.data_name,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_neighbors=args.num_neighbors,
        root=args.root,
        kg_model=args.kg_model,
        kg_batch_size=args.kg_batch_size,
        kg_epochs=args.kg_epochs,
        k_hops=args.k_hops,
        task_type=args.task_type,
        k_shot=args.k_shot,
        num_trails=args.num_trails,
        num_val=args.num_val,
        num_test=args.num_test,
        align_coef=args.align_coef,
        batch_size=args.batch_size,
        lr_task=args.lr_task,
        task_weight_decay=args.task_weight_decay,
        task_epochs=args.task_epochs,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        patience=args.patience,
        log_path=args.log_path,
        checkpoint_dir=args.checkpoint_dir
    )

    if args.config_save_path:
        save_config_to_yaml(config, args.config_save_path)

    return config