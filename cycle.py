"""
cycle.py

TODO Implement Cycle consistency
"""
import torch


def cycle_consistency(dataset1, dataset2):
    N = min(dataset1.frames.shape[0], dataset2.frames.shape[0])
    N = N - N % 4
    nb_frames = N // 4
    print('N      :', N)
    print('Frames :', nb_frames)

    # Split dataset into frame stacks
    stacks1 = torch.split(dataset1.frames[:N], 4, dim=0)
    stacks2 = torch.split(dataset2.frames[:N], 4, dim=0)
