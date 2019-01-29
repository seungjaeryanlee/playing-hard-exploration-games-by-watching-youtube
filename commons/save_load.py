"""Save and load models."""
import torch


def save_models(
    tdc, cmc, tdc_classifier, cmc_classifier, optimizer, path="saves/", prefix="best_"
):
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


def load_models(tdc, cmc, path="saves/", prefix="best_"):
    """
    Load trained models from .pth files.

    Parameters
    ----------
    tdc : nn.Module
        Trained visual embedding network.
    cmc : nn.Module
        Trained audio embedding network.
    path : str
        PATH to save the `.pth` files.
    prefix : str
        Prefix to append to each neural network filename.

    """
    prefix = path + prefix
    tdc.load_state_dict(torch.load("{}tdc.pth".format(prefix)))
    cmc.load_state_dict(torch.load("{}cmc.pth".format(prefix)))
