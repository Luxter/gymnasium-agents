import random

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Set seed {seed}")
