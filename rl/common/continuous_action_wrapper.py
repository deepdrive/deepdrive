import gym


class CombineBoxSpaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CombineBoxSpaceWrapper, self).__init__(env)
        ac_space = env.action_space
        self.denormalizers = None
        if isinstance(ac_space, gym.spaces.Tuple):
            self.denormalizers = []
            box_spaces = [s for s in ac_space.spaces if isinstance(s, gym.spaces.Box)]
            for i, space in enumerate(box_spaces):
                denormalizer = lambda x: ((x + 1) * (space.high - space.low) / 2 + space.low)
                self.denormalizers.append(denormalizer)

    def step(self, action):
        # Denormalize action according to mapping
        denorm_action = []
        for i, denorm in enumerate(self.denormalizers):
            denorm_action.append(denorm(action[i]))
        obz, reward, done, info = self.env.step(denorm_action)
        return obz, reward, done, info

    def reset(self):
        return self.env.reset()