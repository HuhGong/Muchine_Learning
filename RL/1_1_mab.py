# 1_1_mab.py

# mab: Multi-Armed Bandits

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


def greedy(means):
    return np.argmax(means)  # exploitation


def e_greedy(means, e):
    if np.random.rand() < e:
        return np.random.choice(range(len(means)))  # exploration,
    return np.argmax(means)


# rand, radn, choice 함수를 1만번 호출한 결과를 그래프로 그려보세요 (히스토그램)
def show_randoms(bandits):
    n0, n1, n2 = [], [], []
    for _ in range(10000):
        n0.append(np.random.rand())
        n1.append(np.random.randn())
        n2.append(np.random.choice(30))
    plt.subplot(1, 3, 1)
    plt.hist(n0)
    plt.subplot(1, 3, 2)
    plt.hist(n1)
    plt.subplot(1, 3, 3)
    plt.hist(n2)

    plt.show()


def show_casino(bandits, N):
    means = [0] * len(bandits)
    samples = [0] * len(bandits)
    for _ in range(N):
        # select = greedy(means)
        select = e_greedy(means, 0.1)
        # print(select)
        reward = bandits[select] + np.random.randn()
        samples[select] += 1
        ratio = 1 / samples[select]

        means[select] = (1 - ratio) * means[select] + ratio * reward

    # print('mean : ', np.array(means))
    # print('argmax : ', np.argmax(means))
    # print('samples : ', samples)


bandits = [1.0, -2.0, 3.0]
# print(greedy(bandits))
# show_randoms(bandits)
for i in range(10):
    show_casino(bandits, 10000)
