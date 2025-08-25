import torch
import os
import json
import shutil
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(state: Dict[str, Any], filename: str, is_best: bool = False):
    """

    Args:
        state: including model, optimizer, scheduler state_dict
        filename: checkpoint path
        is_best: save the best model
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        torch.save(state, filename)
        logger.info(f"Checkpoint saved to {filename}")

        if is_best:
            best_filename = os.path.join(os.path.dirname(filename), 'model_best.pth')
            shutil.copyfile(filename, best_filename)
            logger.info(f"Best model saved to {best_filename}")

    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filename}: {str(e)}")
        raise


def load_checkpoint(checkpoint_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """

    Args:
        checkpoint_path: checkpoint path
        map_location: cuda or cpu

    Returns:
        state_dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        # check the checkpoint
        if 'state_dict' not in checkpoint:
            logger.warning("Checkpoint does not contain state_dict!")
        if 'epoch' not in checkpoint:
            logger.warning("Checkpoint does not contain epoch information!")

        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
        raise


def save_config(config: Any, config_path: str):
    """
    Save JSON file

    Args:
        config: config object
        config_path: config file path
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = dict(config)

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        logger.info(f"Config saved to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {str(e)}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    load config from JSON file

    Args:
        config_path:

    Returns:
        DotDict
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        logger.info(f"Config loaded from {config_path}")
        return config_dict

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """

    Args:
        checkpoint_dir: checkpoint path

    Returns:
        the latest checkpoint path, if not exist, return None
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and file != 'model_best.pth':
            checkpoint_files.append(file)

    if not checkpoint_files:
        return None

    # ordered in epoch
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 5):
    """

    Args:
        checkpoint_dir: checkpoint path
        keep_last: the number of checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return

    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and file != 'model_best.pth':
            checkpoint_files.append(file)

    if len(checkpoint_files) <= keep_last:
        return

    # ordered in epoch
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # delete old checkpoints
    for file in checkpoint_files[:-keep_last]:
        file_path = os.path.join(checkpoint_dir, file)
        os.remove(file_path)
        logger.info(f"Removed old checkpoint: {file_path}")