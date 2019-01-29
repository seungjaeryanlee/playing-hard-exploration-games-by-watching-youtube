"""Make wrapped standardized environments."""
from .atari_wrappers import make_atari, wrap_deepmind
from .torch_wrappers import wrap_pytorch
from .youtube_wrappers import wrap_youtube


def make_env(embedding_net, ckpts):
    """
    Make wrapped standardized environments.

    Parameters
    ----------
    embedding_net
        TDC network for embedding environment observations.
    ckpts
        Embedded checkpoints for intrinsic rewards.

    """
    env = make_atari("MontezumaRevengeNoFrameskip-v4")
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    env = wrap_youtube(env, embedding_net, ckpts)

    return env
