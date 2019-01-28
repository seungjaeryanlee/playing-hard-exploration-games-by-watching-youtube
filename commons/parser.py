"""
parser.py
"""
import argparse


def get_args(description='YouTube'):
    parser = argparse.ArgumentParser(description)
    # Hyperparameters for pretraining embedding
    # TODO Find correct lambda value from paper
    parser.add_argument('--cmc-lambda', action='store', dest='CMC_LAMBDA',
                        default=1, type=float,
                        help='Weight for combining TDC and CMC loss. Defaults to 1.')
    parser.add_argument('--lr', action='store', dest='LR',
                        default=1e-4, type=float,
                        help='Learning rate for Adam optimizer. Defaults to 1e-4.')
    parser.add_argument('--batch-size', action='store', dest='BATCH_SIZE',
                        default=32, type=int,
                        help='Batch size for pretraining embedding. Defaults to 32.')
    parser.add_argument('--nb-steps', action='store', dest='NB_STEPS',
                        default=200000, type=int,
                        help='Number of training steps for embedding. Defaults to 200000.')

    # Misc. arguments for pretraining embedding
    parser.add_argument('--save-interval', action='store', dest='SAVE_INTERVAL',
                        default=10000, type=int,
                        help='Interval for saving models during pretraining. Defaults to 10000.')
    parser.add_argument('--tsne-interval', action='store', dest='TSNE_INTERVAL',
                        default=1000, type=int,
                        help='Interval for plotting t-SNE during pretraining. Defaults to 1000.')

    # Hyperparameters for training DQN agent
    parser.add_argument('--ckpt-freq', action='store', dest='CKPT_FREQ',
                        default=16, type=int,
                        help='Frequency of checkpoints (N) selected from embedding. Defaults to 16.')
    parser.add_argument('--ckpt-horizon', action='store', dest='CKPT_HORIZON',
                        default=1, type=int,
                        help=' Horizon(Δt) for checkpoints. Defaults to 1.')
    parser.add_argument('--imitation-cutoff', action='store', dest='IMITATION_CUTOFF',
                        default=0.5, type=float,
                        help='Cutoff (α) for giving imitation reward. Defaults to 0.5.')

    args = parser.parse_args()

    return args
