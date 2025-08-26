# checkpoint.py

import torch
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainerCheckpoint:
    """
    标准化的训练检查点数据结构
    """
    epoch: int
    state_dict: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            'epoch': self.epoch,
            'state_dict': self.state_dict,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'config': self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainerCheckpoint':
        """从字典构建 TrainerCheckpoint，自动校验必要字段"""
        missing = []
        for field in ['epoch', 'state_dict', 'optimizer', 'scheduler', 'config']:
            if field not in data:
                missing.append(field)

        if missing:
            raise KeyError(f"TrainerCheckpoint.from_dict: missing keys {missing}")

        return cls(
            epoch=data['epoch'],
            state_dict=data['state_dict'],
            optimizer=data['optimizer'],
            scheduler=data['scheduler'],
            config=data['config']
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: Dict[str, Any],
    filepath: str,
    is_best: bool = False
):
    """
    保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前训练轮数（从1开始或0开始均可）
        config: 配置对象的 __dict__
        filepath: 保存路径
        is_best: 是否是当前最佳模型
    """
    try:
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # 构建标准 checkpoint
        ckpt = TrainerCheckpoint(
            epoch=epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            config=config
        )

        torch.save(ckpt.to_dict(), filepath)
        logger.info(f"Checkpoint saved to {filepath}")

        if is_best:
            best_path = Path(filepath).parent / "model_best.pth"
            torch.save(ckpt.to_dict(), best_path)
            logger.info(f"Best model saved to {best_path}")

    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {str(e)}")
        raise


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    map_location: Optional[str] = None
) -> int:
    """
    加载训练检查点并恢复状态

    Args:
        filepath: 检查点文件路径
        model: 模型对象
        optimizer: 优化器对象（可选）
        scheduler: 调度器对象（可选）
        map_location: 设备映射，如 'cpu', 'cuda'

    Returns:
        int: 恢复的 epoch（可用于设置 start_epoch）

    Raises:
        FileNotFoundError: 文件不存在
        KeyError: 结构不完整
        Exception: 其他加载错误
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    try:
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint_dict = torch.load(filepath, map_location=map_location)
        logger.info(f"Checkpoint loaded from {filepath}")

        # 使用 TrainerCheckpoint 进行结构校验
        ckpt = TrainerCheckpoint.from_dict(checkpoint_dict)

        # 恢复模型
        model.load_state_dict(ckpt.state_dict)

        # 恢复优化器（如果传入）
        if optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt.optimizer)
                logger.debug("Optimizer state restored.")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        # 恢复调度器（如果传入）
        if scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt.scheduler)
                logger.debug("Scheduler state restored.")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        logger.info(f"Successfully resumed from epoch {ckpt.epoch}")
        return ckpt.epoch

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
        raise


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    获取目录中最新的 checkpoint 文件（按 epoch 排序）

    Args:
        checkpoint_dir: 检查点目录

    Returns:
        最新 checkpoint 路径，若无则返回 None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    files = []
    for f in checkpoint_dir.iterdir():
        if f.suffix == '.pth' and f.name != 'model_best.pth' and 'pretrain_epoch' in f.name:
            try:
                epoch = int(f.stem.split('_')[-1])  # 支持 pretrain_epoch_10.pth
                files.append((epoch, f))
            except (ValueError, IndexError):
                continue

    if not files:
        return None

    latest_file = sorted(files, key=lambda x: x[0])[-1][1]
    return str(latest_file)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 5):
    """
    清理旧的 checkpoint，保留最新的 N 个

    Args:
        checkpoint_dir: 检查点目录
        keep_last: 保留最近几个
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    files = []
    for f in checkpoint_dir.iterdir():
        if f.suffix == '.pth' and f.name != 'model_best.pth' and 'pretrain_epoch' in f.name:
            try:
                epoch = int(f.stem.split('_')[-1])
                files.append((epoch, f))
            except:
                continue

    if len(files) <= keep_last:
        return

    # 按 epoch 排序，保留最后 keep_last 个
    files.sort(key=lambda x: x[0])
    for _, f in files[:-keep_last]:
        f.unlink()
        logger.info(f"Removed old checkpoint: {f}")