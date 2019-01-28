import torch.nn as nn


class CMC(nn.Module):
    def __init__(self, in_channels=256, out_channels=1024):
        """
        Embedding network for raw audio.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input audio sample. Defaults to 256.
        out_channels : int
            Number of features in the output layer. Defaults to 1024.
        """
        # Below is a paragraph from the original paper:
        #
        # The audio embedding function, ψ, is as per φ except that it
        # as four, width-8, 1D convolutional layers with (32, 64, 128,
        # 256) channels and 2x max-pooling, and a single width-1024
        # linear layer. The input is a width-137 (6ms) sample of 256
        # frequency channels, calculated using STFT. ReLU-activation and
        # batch-normalization are applied throughout and the embedding
        # vector is l2-normalized.
        #
        # From Section 5: Implementation Details

        # TODO No padding?
        # TODO No residual block?
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, 8, padding=0),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 8, padding=0),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 8, padding=0),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 8, padding=0),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(nn.Linear(512, out_channels))

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
