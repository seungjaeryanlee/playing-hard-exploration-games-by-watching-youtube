#!/usr/bin/env python3
"""Train agent using embedder and embedded checkpoints."""
from typing import Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from commons import load_tdc
from datasets import VideoDataset
from networks import TDC
from wrappers import make_env


def get_checkpoint_loader() -> Any:
    """
    Get dataloder for extracting checkpoints.

    Returns
    -------
    loader : torch.utils.data.DataLoader
        Dataloader for extracting checkpoints.

    """
    filename = "./data/6zXXZvVvTFs"
    trim = (960, 1403)
    crop = (35, 50, 445, 300)
    dataset = VideoDataset(filename, trim, crop, frame_rate=15)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    return loader


def get_checkpoints(tdc: Any, loader: Any) -> List[Any]:
    """
    Extract checkpoints from given dataloader using given TDC network.

    Parameters
    ----------
    tdc : nn.Module
        Video embedding network.
    loader : torch.utils.data.DataLoader
        Dataloader for extracting checkpoints.

    Returns
    -------
    checkpoints : list
        List of embedded checkpoints.

    """
    embed_batches = []
    for _, batch in enumerate(loader):
        embed_batch = tdc(batch.to(device))
        embed_batch = F.normalize(embed_batch).cpu().detach().numpy()
        embed_batches.append(embed_batch)

    return embed_batches


def train_agent(device: Any) -> None:
    """
    Train agent using embedder and embedded checkpoints.

    TODO Fix docstrings once finished.

    """
    # Load embedded network
    tdc = TDC().to(device)
    load_tdc(tdc)

    # Create checkpoints
    loader = get_checkpoint_loader()
    checkpoints: List[torch.Tensor] = get_checkpoints(tdc, loader)
    print(checkpoints)

    # Create environment
    env = make_env(tdc, checkpoints)

    # TODO Temporarily added to disable flake8 error
    print(env)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_agent(device)
