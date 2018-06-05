import gym


class CombineBoxSpaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CombineBoxSpaceWrapper, self).__init__(env)
        ac_space = env.action_space
        self.denormalizers = None
        if isinstance(ac_space, gym.spaces.Tuple):
            self.denormalizers = []
            box_spaces = [s for s in ac_space.spaces if isinstance(s, gym.spaces.Box)]
            total_dims = 0
            for i, space in enumerate(box_spaces):
                if len(space.shape) > 1 or space.shape[0] > 1:
                    raise NotImplementedError('Multi-dimensional box spaces not yet supported - need to flatten / separate')
                else:
                    total_dims += 1

                def denormalizer(x):
                    ret = ((x + 1) * (space.high[0] - space.low[0]) / 2 + space.low[0])
                    return ret
                self.denormalizers.append(denormalizer)
            self.action_space = gym.spaces.Box(-1, 1, shape=(total_dims,))

    def step(self, action):
        # Denormalize action according to mapping
        denorm_action = []
        for i, denorm in enumerate(self.denormalizers):
            denorm_action.append([denorm(action[i])])
        obz, reward, done, info = self.env.step(denorm_action)
        return obz, reward, done, info

    def reset(self):
        return self.env.reset()
