import json
import os
import argparse
from utils.logger import create_logger
from typing import Dict, Any, Optional


def base_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='./datasets')
    parser.add_argument("--pretrain_data_names", type=str, nargs="+",
                        default=["ogbn-arxiv", "AmazonProducts", "Reddit", "FB15k_237", "PCBA", "PPI"])
    config = parser.parse_args()
    return config


def save_config(config: Any, config_path: str, logger):
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


def load_config(config_path: str, logger) -> Dict[str, Any]:
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


def list2str(l):
    s = ""
    for e in l[:-1]:
        s += f"{e}_"
    s += f"{l[-1]}"
    return s


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        super().__init__()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value