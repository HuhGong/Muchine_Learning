# 1_2_mab_class.py
import numpy as np

# mab: Multi-Armed Bandits

import numpy as np

np.set_printoptions(precision=3)


# 퀴즈
# 앞에서 만든 코드를 클래스 버전으로 수정하세요

class SlotMachine:
    def __init__(self, reward):
        self.mean = 0
        self.sample = 0
        self.reward = reward

    def pull(self):
        return self.reward + np.random.randn()

    def update(self, reward):
        self.sample += 1
        ratio = 1 / self.sample

        self.mean = (1 - ratio) * self.mean + ratio * reward


class Casino:
    def __init__(self, bendits, e):
        self.bandits = bandits
        self.e = e


    def e_greedy(self):
        if np.random.rand() < self.e:
            return np.random.choice(range(len(self.bandits)))  # exploration

        return np.argmax([s.mean for s in self.bandits])

    def show(self, N):
        for _ in range(N):
            select = self.e_greedy()
            reward = self.bandits[select].pull()
            self.bandits[select].update(reward)
        print('means : ', np.array([s.mean for s in self.bandits]))


bandits = [SlotMachine(1.0), SlotMachine(-2.0), SlotMachine(3.0)]
kangwon = Casino(bandits, e=0.1)
kangwon.show(10000)
