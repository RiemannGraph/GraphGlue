# configs.py

import argparse
import yaml
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class PretrainConfig:
    """
    预训练配置类（使用 dataclass 管理）
    """
    # Data
    num_neighbors: List[int]
    k_hops: int
    pretrain_single_graph_data: List[str]
    pretrain_multi_graph_data: List[str]
    batch_size: int = 128
    num_workers: int = 8

    # Training
    pretrain_epochs: int = 100
    lr_pretrain: float = 1e-3
    pretrain_weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 10
    resume_checkpoint: bool = False

    # Loss & Graph
    knn: int = 5

    # Paths
    log_path: str = "logs/pretrain.log"
    checkpoint_dir: str = "checkpoints/pretrain/"


def get_config_parser() -> argparse.ArgumentParser:
    """
    创建 argparse 解析器
    """
    parser = argparse.ArgumentParser(description="Graph Pretraining Configuration")

    # Data
    parser.add_argument('--num_neighbors', type=int, nargs='+', required=True,
                        help='每跳采样的邻居数量，例如: --num_neighbors 10 10')
    parser.add_argument('--k_hops', type=int, required=True,
                        help='子图采样跳数，必须 <= len(num_neighbors)')
    parser.add_argument('--pretrain_single_graph_data', type=str, nargs='+', default=[],
                        help='单图预训练数据集列表，例如: pubmed cora')
    parser.add_argument('--pretrain_multi_graph_data', type=str, nargs='+', default=[],
                        help='多图预训练数据集列表，例如: nci1 mutag')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loading (default: 128)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading (default: 8)')

    # Training
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Total pretrain epochs (default: 100)')
    parser.add_argument('--lr_pretrain', type=float, default=1e-3,
                        help='Learning rate for pretraining (default: 0.001)')
    parser.add_argument('--pretrain_weight_decay', type=float, default=1e-5,
                        help='Weight decay for Adam optimizer (default: 1e-5)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches (default: 10)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume_checkpoint', action='store_true',
                        help='Whether to resume from latest checkpoint')

    # Loss & Graph
    parser.add_argument('--knn', type=int, default=5,
                        help='KNN graph connections for inter-graph loss (default: 5)')

    # Paths
    parser.add_argument('--log_path', type=str, default="logs/pretrain.log",
                        help='Path to log file (default: logs/pretrain.log)')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/pretrain/",
                        help='Directory to save checkpoints (default: checkpoints/pretrain/)')

    # Config IO
    parser.add_argument('--config_save_path', type=str, default=None,
                        help='Path to save current config as YAML (optional)')
    parser.add_argument('--config_load_path', type=str, default=None,
                        help='Path to load config from YAML (optional, will override cmd args)')

    return parser


def save_config_to_yaml(config: PretrainConfig, filepath: str):
    """
    将 dataclass 配置保存为 YAML 文件
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(asdict(config), f, default_flow_style=False, indent=2, sort_keys=False)
    print(f"Config saved to {filepath}")


def load_config_from_yaml(filepath: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_config() -> PretrainConfig:
    """
    主函数：解析命令行或 YAML 配置，返回 PretrainConfig 实例
    """
    parser = get_config_parser()
    args = parser.parse_args()

    # 如果提供了 YAML 配置文件，先加载它
    if args.config_load_path:
        print(f"Loading config from YAML: {args.config_load_path}")
        yaml_config = load_config_from_yaml(args.config_load_path)

        # 将 yaml_config 转换为 argparse 可接受的格式（flatten）
        # 注意：argparse 会覆盖 yaml 中的值（优先级更高）
        # 所以我们先用 yaml 值作为默认值，再用 argparse 覆盖
        for key, value in yaml_config.items():
            if hasattr(args, key):
                # 如果 argparse 没有传这个参数，就用 yaml 的值
                if not any(opt_str in str(sys.argv) for opt_str in [f'--{key}', f'-{key}']):
                    setattr(args, key, value)

    # 构建 PretrainConfig
    config = PretrainConfig(
        num_neighbors=args.num_neighbors,
        k_hops=args.k_hops,
        pretrain_single_graph_data=args.pretrain_single_graph_data,
        pretrain_multi_graph_data=args.pretrain_multi_graph_data,
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

    # 保存配置（如果指定了路径）
    if args.config_save_path:
        save_config_to_yaml(config, args.config_save_path)

    return config