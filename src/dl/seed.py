"""
Deterministic random seed control and device selection for DL experiments.

Provides set_seed() to ensure reproducible execution across Python,
NumPy, PyTorch, and CUDA/cuDNN.
"""

import os
import random
from typing import Optional

import numpy as np
import torch

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Configure random seeds across all libraries for deterministic execution.

    Sets seeds for:
    - Python built-in `random`
    - NumPy `np.random`
    - PyTorch CPU `torch.manual_seed`
    - PyTorch CUDA `torch.cuda.manual_seed_all`
    - Environment variable `PYTHONHASHSEED`
    - cuDNN deterministic algorithms

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Deterministic seed set to {seed} across Python, NumPy, and PyTorch.")


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Get the primary computing device (CUDA or CPU).

    Args:
        preferred: Optional preferred device string (e.g. 'cuda', 'cuda:0', 'cpu').
            If None or not available, automatically defaults to CUDA when available.

    Returns:
        torch.device instance.
    """
    if preferred is not None:
        try:
            device = torch.device(preferred)
            if device.type == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    f"CUDA requested ('{preferred}') but CUDA is not available. Falling back to CPU."
                )
                return torch.device("cpu")
            return device
        except Exception as e:
            logger.warning(f"Invalid device '{preferred}': {e}. Falling back to auto-detection.")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA is available. Using GPU: {device_name}")
        return torch.device("cuda:0")

    logger.info("CUDA is not available. Using CPU.")
    return torch.device("cpu")
