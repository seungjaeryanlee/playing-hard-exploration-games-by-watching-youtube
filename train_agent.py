#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from commons import load_models
from datasets import VideoDataset
from networks import CMC, TDC
from wrappers import make_env


def get_checkpoint_loader():
    filename = "./data/6zXXZvVvTFs"
    trim = (960, 1403)
    crop = (35, 50, 445, 300)
    dataset = VideoDataset(filename, trim, crop, frame_rate=15)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    return loader

def get_checkpoints(tdc, loader):
    embed_batches = []
    for _, batch in enumerate(loader):
        stack_batch, _ = batch
        embed_batch = tdc(stack_batch.to(device))
        embed_batch = F.normalize(embed_batch).cpu().detach().numpy()
        embed_batches.append(embed_batch)
        break

    return embed_batches

def train_agent(device):
    # Load embedded network
    tdc = TDC().to(device)
    cmc = CMC().to(device)
    load_models(tdc, cmc)

    # Create checkpoints
    loader = get_checkpoint_loader()
    checkpoints = get_checkpoints(tdc, loader)
    print(checkpoints)

    # Create environment
    env = make_env((tdc, cmc), checkpoints)

    # TODO Temporarily added to disable flake8 error
    print(env)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_agent(device)
