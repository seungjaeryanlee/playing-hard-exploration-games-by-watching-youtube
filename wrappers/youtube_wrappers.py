"""Environment wrapper for using YouTube checkpoints."""
import gym

DELTA_T = 1
ALPHA = 0.5


class YouTubeWrapper(gym.Wrapper):
    """
    Wrap environment to include imitation reward.

    Environment wrapper that gives imitation reward if the observation
    is sufficiently similar to checkpoints.

    """

    def __init__(self, env, embedding_net, ckpts):
        gym.Wrapper.__init__(self, env)
        self.embedding_net = embedding_net
        self.ckpts = ckpts
        self.ckpt_first = 0
        self.reset = self.env.reset()

    def step(self, action):  # noqa: D102
        ob, reward, done, info = self.env.step(action)

        # Compute imitation reward
        imitation_reward = 0
        for ckpt_i in range(self.ckpt_first, self.ckpt_first + DELTA_T + 1):
            if self.embedding_net(ob) * self.ckpts[ckpt_i] > ALPHA:
                imitation_reward = 0.5
                self.ckpt_first = ckpt_i
                break

        return ob, reward + imitation_reward, done, info


def wrap_youtube(env, embedding_net, ckpts):
    """Wrap environment to use YouTube checkpoints."""
    env = YouTubeWrapper(env, embedding_net, ckpts)

    return env
