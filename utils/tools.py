import torch
import numpy as np
import random
from datetime import timedelta


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def format_time(t):
    return str(timedelta(seconds=int(t)))