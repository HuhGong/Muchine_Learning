# 3_3_q_network.py

import gym
import numpy as np
import util
import matplotlib.pyplot as plt
import keras


def random_argmax(rewards):
    r_max = np.max(rewards)
    indices = np.nonzero(rewards == r_max)

    return np.random.choice(indices[0])


def e_greedy(i, env, actions):
    # e = 1 / (i // 100 + 1)
    e = 0.1 / (i + 1)
    # return np.random.randint(4) if np.random.rand() < e else random_argmax(actions)
    if np.random.rand() < e:
        return env.action_space.sample()
    return random_argmax(actions)


def random_noise(i, env, actions):
    # values = actions + np.random.rand(len(actions))
    # values = actions + np.random.randn(len(actions))       # 잘 안나옴 성공 : 226.0 0.113
    values = actions + np.random.randn(env.action_space.n) / (i + 1)
    return np.argmax(values)


def make_onehot(state):
    z = np.zeros(16)
    z[state] = 1
    return z.reshape(1, -1)


def q_network(loop):
    env = gym.make('FrozenLake-v1', is_slippery=False)  # , stochastic world
    # env = gym.make('FrozenLake-v1', is_slippery=False)  # , render_mode='human'
    # q_table = np.zeros((16, 4))
    model = keras.Sequential([
        keras.layers.Input(shape=(16,)),
        keras.layers.Dense(4, use_bias=False)
    ])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mean_squared_error)

    success, discounted = 0, 0.9
    results = []
    for i in range(loop):
        state, _ = env.reset()

        done = False
        while not done:
            p = model.predict(make_onehot(state), verbose=2)
            action = random_noise(i, env, actions=p[0])
            next_state, reward, done, _, _ = env.step(action)
            if done:
                p[0, action] = reward
            else:
                p_next = model.predict(make_onehot(next_state), verbose=0)
                p[0, action] = reward + discounted * np.max(p_next[0])

            model.fit(make_onehot(state), p, epochs=1, verbose=2)
            state = next_state

        success += reward
        results.append(reward)
        if i % 10 == 0:
            print(i)

    # util.draw_q_table(q_table)
    print('성공 :', int(success), success / loop)

    print('count : ', i + 1, 'goal_times', success)
    plt.plot(range(len(results)), results)

    plt.show()


q_network(2000)
