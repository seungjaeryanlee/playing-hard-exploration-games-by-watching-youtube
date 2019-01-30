"""Train agent using embedder and embedded checkpoints."""
from typing import Any, List

import torch

from commons import load_models
from networks import CMC, TDC
from wrappers import make_env


def train_agent(device: Any) -> None:
    """
    Train agent using embedder and embedded checkpoints.

    TODO Fix docstrings once finished.

    """
    # Load embedded network
    tdc = TDC().to(device)
    cmc = CMC().to(device)
    load_models(tdc, cmc)

    # TODO Create checkpoints
    checkpoints: List[torch.Tensor] = []

    # Create environment
    env = make_env((tdc, cmc), checkpoints)

    # TODO Temporarily added to disable flake8 error
    print(env)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_agent(device)
