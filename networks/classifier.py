"""Classifier for training TDC and CMC embedder."""
import torch.nn as nn


class Classifier(nn.Module):
    """
    Classifier network used for both TDC and CMC.

    Parameters
    ----------
    in_channels : int
        Number of features in the input layer. Should match the output
        of TDC and CMC. Defaults to 1024.
    out_channels : int
        Number of features in the output layer. Should match the number
        of classes. Defaults to 6.

    """

    def __init__(self, in_channels=1024, out_channels=6):
        # Below is a paragraph from the original paper:
        #
        # The same shallow network architecture, τ , is used for both temporal
        # and cross-modal classification. Both input vectors are combined by
        # element-wise multiplication, with the result fed into a 2-layer MLP
        # with widths (1024, 6) and ReLU non-linearity in between. A
        # visualization of these networks and their interaction is provided in
        # Figure 3. Note that although τ_td and τ_cm share the same
        # architecture, they are operating on two different problems and
        # therefore maintain separate sets of weights.
        #
        # From Section 5: Implementation Details
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 1024), nn.ReLU(), nn.Linear(1024, out_channels)
        )

    def forward(self, x):  # noqa: D102
        out = self.layers(x)
        return out
