import gym

DELTA_T = 1
ALPHA = 0.5


class YouTubeWrapper(gym.Wrapper):
    def __init__(self, env, embedding_net, ckpts):
        gym.Wrapper.__init__(self, env)
        self.embedding_net = embedding_net
        self.ckpts = ckpts
        self.ckpt_first = 0
        self.reset = self.env.reset()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        # Compute imitation reward
        imitation_reward = 0
        for ckpt_i in range(ckpt_first, ckpt_first + DELTA_T + 1):
            if embedding_net(ob) * ckpts[ckpt_i] > ALPHA:
                imitation_reward = 0.5
                ckpt_first = ckpt_i
                break

        return ob, reward + imitation_reward, done, info


def wrap_youtube(env, embedding_net, ckpts):
    """
    Wrap environment to use YouTube checkpoints.
    """
    env = YouTubeWrapper(env, embedding_net, ckpts)

    return env
