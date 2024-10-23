# 2_3_mc_td.py
import numpy as np
# 사용안함
np.set_printoptions(precision=2)  # 소수점 자리수 조절

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def mont_carlo(loop, alpha):
    class GridEnvironment:
        def __init__(self, size=4):
            self.row = 0
            self.col = 0
            self.size = size

        def reset(self):
            self.row = 0
            self.col = 0

            return self.row, self.col

        def step(self, action):
            if action == LEFT and self.col > 0:
                self.col -= 1
            elif action == RIGHT and self.col < self.size - 1:
                self.col += 1
            elif action == UP and self.row > 0:
                self.row -= 1
            elif action == DOWN and self.row < self.size - 1:
                self.row += 1

            return (self.row, self.col), -1, self.is_done()

        def is_done(self):
            return self.row == self.size - 1 and self.col == self.size - 1

    def select_action():
        # coin = np.random.rand()
        # return int(coin / 0.25)
        return np.random.choice([LEFT, DOWN, RIGHT, UP])
        # return np.random.randint(4)

    env = GridEnvironment()
    grid = np.zeros([env.size, env.size])
    for i in range(loop):
        state = env.reset()

        episode = []
        done = False
        while not done:
            action = select_action()
            next_state, reward, done = env.step(action)
            # print(state)
            state = next_state

            episode.append((state, reward))
            state = next_state

        # print(i, episode)

        cum_reward = 0
        for state, reward in episode[::-1]:
            grid[state] += alpha * (cum_reward - grid[state])
            cum_reward += reward
    print(grid)


def temporal_difference(loop, alpha):
    class GridEnvironment:
        def __init__(self, size=4):
            self.row = 0
            self.col = 0
            self.size = size

        def reset(self):
            self.row = 0
            self.col = 0

            return self.row, self.col

        def step(self, action):
            if action == LEFT and self.col > 0:
                self.col -= 1
            elif action == RIGHT and self.col < self.size - 1:
                self.col += 1
            elif action == UP and self.row > 0:
                self.row -= 1
            elif action == DOWN and self.row < self.size - 1:
                self.row += 1

            return (self.row, self.col), -1, self.is_done()

        def is_done(self):
            return self.row == self.size - 1 and self.col == self.size - 1

    def select_action():
        return np.random.choice([LEFT, DOWN, RIGHT, UP])

    env = GridEnvironment()
    grid = np.zeros([env.size, env.size])
    for i in range(loop):
        state = env.reset()

        done = False
        while not done:
            action = select_action()
            next_state, reward, done = env.step(action)
            # 요기에서 업데이트 할 것을 채워주세요
            grid[state] += alpha * (reward + grid[next_state] - grid[state])
            state = next_state

        # print(i, episode)

        # cum_reward = 0
        # for state, reward in episode[::-1]:
        #     grid[state] += alpha * (cum_reward - grid[state])
        #     cum_reward += reward
    print(grid)


# mont_carlo(100, 0.001)
temporal_difference(10000, alpha=0.1)
