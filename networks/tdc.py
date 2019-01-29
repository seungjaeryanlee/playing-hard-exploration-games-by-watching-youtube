"""TDC network for embedding video."""
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A convolutional block with residual connections.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input.
    out_channels : int
        Number of features in the output.

    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):  # noqa: D102
        out = self.layers(x) + x
        return self.final_relu(out)


class TDC(nn.Module):
    """
    Embedding network for video frames.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input framestack. Defaults to 12.
    out_channels : int
        Number of features in the output layer. Defaults to 1024.

    """

    def __init__(self, in_channels=12, out_channels=1024):
        # Below is a paragraph from the original paper:
        #
        # The visual embedding function, φ, is composed of three spatial,
        # padded, 3x3 convolutional layers with (32, 64, 64) channels and 2x2
        # max-pooling, followed by three residual-connected blocks with 64
        # channels and no down-sampling. Each layer is ReLU-activated and batch
        # -normalized, and the output fed into a 2-layer 1024-wide MLP. The
        # network input is a 128x128x3x4 tensor constructed by random spatial
        # cropping of a stack of four consecutive 140x140 RGB images, sampled
        # from our dataset. The final embedding vector is l2-normalized.
        #
        # From Section 5: Implementation Details
        # TODO Stride instead of max pooling?
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16384, 1024), nn.Linear(1024, out_channels)
        )

    def forward(self, x):  # noqa: D102
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
