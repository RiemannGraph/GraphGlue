import sys
import argparse
import yaml
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class AdaptionConfig:
    # Data
    data_name: str
    pretrained_checkpoint: str
    num_neighbors: List[int]
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
    log_path: str = "logs/task.log"
    checkpoint_dir: str = "checkpoints/task/"