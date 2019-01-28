#!/usr/bin/env python3
"""
pretrain_embedding.py
"""
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import wandb

from commons import save_models, get_args
from datasets import TDCCMCDataset, LazyTDCCMCDataset
from networks import Classifier, TDC, CMC
from tsne import get_tsne_loaders, plot_tsne


def pretrain_embeddings(
    loader,
    tdc,
    cmc,
    tdc_classifier,
    cmc_classifier,
    optimizer,
    tsne_loaders,
    device,
    CMC_LAMBDA,
    NB_STEPS,
    SAVE_INTERVAL,
    TSNE_INTERVAL,
):
    """
    Train the embedding networks and classifiers for TDC and CMC.

    Parameters
    ----------
    args : dict
        Dictionary of hyperparmeters.
    """
    # TODO Add .eval() somewhere (t-SNE, Cycle consistency, DQN training)
    tdc.train()
    cmc.train()
    tdc_classifier.train()
    cmc_classifier.train()

    for i in range(1, NB_STEPS + 1):
        start = time.time()
        # Get data from DataLoader
        stack_v, stack_w, sample_a, tdc_label, cmc_label = next(iter(loader))
        stack_v = stack_v.to(device)
        stack_w = stack_w.to(device)
        sample_a = sample_a.to(device)
        tdc_label = tdc_label.to(device)
        cmc_label = cmc_label.to(device)

        # Embed and normalize data
        # TODO Check if F.normalize() is correct
        embed_v = F.normalize(tdc(stack_v))
        embed_w = F.normalize(tdc(stack_w))
        embed_a = F.normalize(cmc(sample_a))

        # Predict with TDC and CMC
        tdc_prediction = tdc_classifier(embed_v * embed_w)
        cmc_prediction = cmc_classifier(embed_v * embed_a)

        # Compute Loss
        # F.cross_entropy expects target to be shaped torch.Size([BATCH_SIZE])
        tdc_loss = F.cross_entropy(tdc_prediction, tdc_label.squeeze())
        cmc_loss = F.cross_entropy(cmc_prediction, cmc_label.squeeze())
        loss = tdc_loss + CMC_LAMBDA * cmc_loss

        # Backprop Loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute epoch duration and epochs per second
        end = time.time()
        duration = end - start
        epoch_per_sec = 1 / duration

        # Log to console and wandb
        print(
            "[pretrain] {}/{}\tLoss {:.6f}\t{:2.2f} seconds".format(
                i, NB_STEPS, loss, duration
            )
        )
        wandb.log(
            {"Loss": loss, "Duration": duration, "Epochs per second": epoch_per_sec}
        )

        # Save model periodically
        if i > 0 and i % SAVE_INTERVAL == 0:
            save_start_t = time.time()
            save_models(
                tdc,
                cmc,
                tdc_classifier,
                cmc_classifier,
                optimizer,
                path=wandb.run.dir,
                prefix="best_",
            )
            save_end_t = time.time()
            print(
                "[pretrain] Saved checkpoint (Duration: {}s)".format(
                    save_end_t - save_start_t
                )
            )

        # Plot t-SNE periodically
        if i % TSNE_INTERVAL == 0:
            tsne_start_t = time.time()
            tdc.eval()
            cmc.eval()
            plot_tsne(tsne_loaders, tdc, cmc, device)
            tdc.train()
            cmc.train()
            tsne_end_t = time.time()
            print(
                "[pretrain] Updated t-SNE figure (Duration: {}s)".format(
                    tsne_end_t - tsne_start_t
                )
            )


if __name__ == "__main__":
    # Parse arguments
    args = get_args()

    # Setup wandb
    wandb.init(project="youtube")
    wandb.config.update(args)
    print("[pretrain] Wandb setup complete.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = TDCCMCDataset(
        filenames=["./data/6zXXZvVvTFs", "./data/2AYaxTiWKoY", "./data/sYbBgkP9aMo"],
        trims=[(960, 9960), (550, 9550), (1, 9901)],
        crops=[(35, 50, 445, 300), (0, 13, 640, 335), (5, 22, 475, 341)],
    )
    loader = DataLoader(
        dataset, batch_size=args.BATCH_SIZE, num_workers=0, pin_memory=True
    )
    print("[pretrain] Dataset and DataLoader ready.")

    tdc = TDC().to(device)
    cmc = CMC().to(device)
    tdc_classifier = Classifier().to(device)
    cmc_classifier = Classifier().to(device)
    print("[pretrain] Neural networks initialized.")

    # Initialize Optimizer
    optim_params = (
        list(tdc.parameters())
        + list(cmc.parameters())
        + list(tdc_classifier.parameters())
        + list(cmc_classifier.parameters())
    )
    optimizer = optim.Adam(optim_params, lr=args.LR)
    print("[pretrain] Optimizer initialized.")

    # Setup t-SNE datasets
    tsne_filenames = [
        "./data/6zXXZvVvTFs",
        "./data/2AYaxTiWKoY",
        "./data/sYbBgkP9aMo",
        "./data/pF6xCZA72o0",
    ]
    tsne_trims = [(960, 1403), (550, 1515), (1, 331), (1465, 2201)]
    tsne_crops = [
        (35, 50, 445, 300),
        (0, 13, 640, 335),
        (5, 22, 475, 341),
        (20, 3, 620, 360),
    ]
    tsne_loaders = get_tsne_loaders(tsne_filenames, tsne_trims, tsne_crops)
    print("[pretrain] Setup t-SNE datasets")

    # Pretrain Embeddings
    print("[pretrain] Begin training.")
    pretrain_embeddings(
        loader,
        tdc,
        cmc,
        tdc_classifier,
        cmc_classifier,
        optimizer,
        tsne_loaders,
        device,
        args.CMC_LAMBDA,
        args.NB_STEPS,
        args.SAVE_INTERVAL,
        args.TSNE_INTERVAL,
    )
    print("[pretrain] Finished training.")

    # Save Model
    save_models(tdc, cmc, tdc_classifier, cmc_classifier, optimizer)
    print("[pretrain] Saved embedding successfully!")
