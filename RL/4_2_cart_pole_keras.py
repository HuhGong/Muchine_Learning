# 4_2_cart_pole_keras.py

import gym
import numpy as np
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


def q_network(loop):
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1')

    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(2, use_bias=False)
    ])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mean_squared_error)

    discounted = 0.9

    results = []
    for i in range(loop):
        state, _ = env.reset()
        print(state)
        # exit()
        done, stand = False, 0

        while not done:
            p = model.predict(state[np.newaxis], verbose=0)
            action = random_noise(i, env, actions=p[0])
            next_state, reward, done, _, _ = env.step(action)
            if done:
                p[0, action] = -100
            else:
                p_next = model.predict(next_state[np.newaxis], verbose=0)
                p[0, action] = reward + discounted * np.max(p_next[0])

            model.fit(state[np.newaxis], p, epochs=1, verbose=0)
            state = next_state
            stand += reward
        print(i, stand)
        results.append(stand)
        if i % 10 == 0:
            print(i)

    # util.draw_q_table(q_table)

    print('count : ', i + 1, 'goal_times', results)
    plt.plot(range(len(results)), results)

    plt.show()


def q_network_sigmoid(loop):
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1')

    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        # (?, 1) = (?, 4) @ (4, 1)
        keras.layers.Dense(1, use_bias=False, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy)

    discounted = 0.99

    results = []
    for i in range(loop):
        state, _ = env.reset()
        # print(state)
        # exit()
        done, stand = False, 0

        while not done:
            p = model.predict(state[np.newaxis], verbose=0)
            action = int(p[0, 0] < 0.5)
            next_state, reward, done, _, _ = env.step(action)

            if done:
                p[0, 0] = -70
            else:
                p_next = model.predict(next_state[np.newaxis], verbose=0)
                # exit()
                p[0, 0] = reward + discounted * np.max(p_next[0])

            model.fit(state[np.newaxis], p, epochs=1, verbose=0)
            state = next_state
            stand += reward
        print(i, stand)
        results.append(reward)
        if i % 10 == 0:
            print(i)

    # util.draw_q_table(q_table)

    print('count : ', i + 1, 'goal_times', results)
    plt.plot(range(len(results)), np.mean(results))

    plt.show()


# 퀴즈
# 기존 버전을 로지스틱 리그레션 버전으로 수정하세요

# q_network(20)
q_network_sigmoid(20)
