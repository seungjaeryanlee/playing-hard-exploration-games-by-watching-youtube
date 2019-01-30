"""Check cycle consistency to test embedding quality."""
from typing import Any

import torch


def cycle_consistency(dataset1: Any, dataset2: Any) -> None:
    """
    Check cycle consistency of two datasets to test embedding quality.

    Parameters
    ----------
    dataset1
        Dataset of video to check cycle consistency of.
    dataset2
        Dataset of video to check cycle consistency of.

    """
    N = min(dataset1.frames.shape[0], dataset2.frames.shape[0])
    N = N - N % 4
    nb_frames = N // 4
    print("N      :", N)
    print("Frames :", nb_frames)

    # Split dataset into frame stacks
    stacks1 = torch.split(dataset1.frames[:N], 4, dim=0)
    stacks2 = torch.split(dataset2.frames[:N], 4, dim=0)

    # TODO Temporarily added to disable flake8 error
    print(stacks1)
    print(stacks2)
