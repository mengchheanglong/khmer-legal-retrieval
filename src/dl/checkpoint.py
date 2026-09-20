"""
Checkpoint management for deep learning experiments.

Implements atomic saving and loading of model weights, optimizer states,
learning rate schedulers, epoch counters, and complete RNG states
to enable exact resumption of training.
"""

import os
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import Optimizer

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def save_checkpoint(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_val_mrr: float = 0.0,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    """Atomically save a model checkpoint with complete training state.

    Args:
        checkpoint_path: Target path for the checkpoint (e.g. results/checkpoints/best.pt).
        model: PyTorch model instance.
        optimizer: Optional optimizer instance.
        scheduler: Optional learning rate scheduler instance.
        epoch: Current completed epoch number.
        best_val_mrr: Best validation MRR@10 achieved so far.
        config: Optional configuration dictionary.

    Returns:
        Path to the saved checkpoint file.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    state = {
        "epoch": epoch,
        "best_val_mrr": best_val_mrr,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng_state,
        "config": config or {},
    }

    # Atomic write pattern: save to temporary file, then replace
    torch.save(state, temp_path)
    if path.exists():
        path.unlink()
    temp_path.rename(path)

    logger.info(
        f"Checkpoint saved atomically -> {path.name} "
        f"(Epoch {epoch}, Best val MRR: {best_val_mrr:.4f})"
    )
    return path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore training states.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: PyTorch model instance into which weights will be loaded.
        optimizer: Optional optimizer into which state will be restored.
        scheduler: Optional scheduler into which state will be restored.
        restore_rng: Whether to restore random number generator states.
        device: Device to map weights onto (default: current model device).

    Returns:
        Metadata dictionary containing 'epoch', 'best_val_mrr', and 'config'.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    map_location = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # 1. Restore model parameters
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Restore optimizer state if provided
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 3. Restore scheduler state if provided
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 4. Restore RNG state if requested
    if restore_rng and "rng_state" in checkpoint:
        rng_state = checkpoint["rng_state"]
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            try:
                t_state = rng_state["torch"]
                if isinstance(t_state, torch.Tensor) and t_state.dtype != torch.uint8:
                    t_state = t_state.to(torch.uint8)
                elif not isinstance(t_state, torch.Tensor):
                    t_state = torch.tensor(t_state, dtype=torch.uint8)
                torch.set_rng_state(t_state)
            except Exception as e:
                logger.warning(f"Could not restore torch RNG state: {e}")
        if torch.cuda.is_available() and "cuda" in rng_state and rng_state["cuda"]:
            try:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            except Exception as e:
                logger.warning(f"Could not restore CUDA RNG state: {e}")

    epoch = checkpoint.get("epoch", 0)
    best_val_mrr = checkpoint.get("best_val_mrr", 0.0)
    logger.info(f"Loaded checkpoint from {path.name} (resumed at epoch {epoch + 1})")

    return {
        "epoch": epoch,
        "best_val_mrr": best_val_mrr,
        "config": checkpoint.get("config", {}),
    }
