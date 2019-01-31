"""Save and load models."""
from typing import Any

import torch


def save_models(
    tdc: Any,
    cmc: Any,
    tdc_classifier: Any,
    cmc_classifier: Any,
    optimizer: Any,
    path: str = "saves/",
    prefix: str = "best_",
) -> None:
    """
    Save trained models to .pth files.

    Parameters
    ----------
    tdc : nn.Module
        Trained visual embedding network.
    cmc : nn.Module
        Trained audio embedding network.
    tdc_classifier : nn.Module
        Trained TDC classifier network.
    cmc_classifier : nn.Module
        Trained CMC classifier network.
    optimizer
        Optimizer used to train these neural networks.
    path : str
        PATH to save the `.pth` files.
    prefix : str
        Prefix to append to each neural network filename.

    """
    prefix = path + prefix
    torch.save(tdc.state_dict(), "{}tdc.pth".format(prefix))
    torch.save(cmc.state_dict(), "{}cmc.pth".format(prefix))
    torch.save(tdc_classifier.state_dict(), "{}tdc_cls.pth".format(prefix))
    torch.save(cmc_classifier.state_dict(), "{}cmc_cls.pth".format(prefix))
    torch.save(optimizer.state_dict(), "{}optim.pth".format(prefix))


def load_models(
    tdc: Any, cmc: Any, path: str = "saves/", prefix: str = "best_"
) -> None:
    """
    Load trained models from .pth files.

    Parameters
    ----------
    tdc : nn.Module
        Visual embedding network to load parameters to.
    cmc : nn.Module
        Audio embedding network to load parameters to.
    path : str
        PATH to load the `.pth` files from.
    prefix : str
        Prefix of each neural network filename.

    """
    prefix = path + prefix
    tdc.load_state_dict(torch.load("{}tdc.pth".format(prefix)))
    cmc.load_state_dict(torch.load("{}cmc.pth".format(prefix)))


def load_tdc(tdc: Any, path: str = "saves/", prefix: str = "best_") -> None:
    """
    Load trained TDC network.

    Parameters
    ----------
    tdc : nn.Module
        Visual embedding network to load parameters to.
    path : str
        PATH to load the `.pth` files from.
    prefix : str
        Prefix of each neural network filename.

    """
    prefix = path + prefix
    tdc.load_state_dict(torch.load("{}tdc.pth".format(prefix)))
