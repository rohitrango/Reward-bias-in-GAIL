import gym


class MiniGridGailWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.state = self.env.reset()
        print("initialised")
    
    # def reward(self, rew):
    #     # modify rew
    #     # print(self.env.env.env.env.actions[0])
    #     print("reward called")
    #     return 2.0

    def step(self, actions):
        # print("step called")
        self.state, r, gameOver, info = self.env.step(actions)

        return self.state, r, gameOver, info

    def reset(self):
        # print("reset called")
        self.state = self.env.reset()
        # print("reset done")
        # print("reset done: ", self.state)
        return self.state