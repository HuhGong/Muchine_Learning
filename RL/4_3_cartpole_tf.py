# 4_3_cartpole_tf.py

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import keras

tf.disable_v2_behavior()


def random_noise(i, env, actions):
    # values = actions + np.random.rand(len(actions))
    # values = actions + np.random.randn(len(actions))       # 잘 안나옴 성공 : 226.0 0.113
    values = actions + np.random.randn(env.action_space.n) / (i + 1)
    return np.argmax(values)


def q_network_sigmoid(loop):
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1')

    x = tf.placeholder(tf.float32, shape=(1, 4))  # state
    y = tf.placeholder(tf.float32, shape=(1, 1))  # action

    w = tf.Variable(tf.random_uniform([4, 1]))
    # b = tf.Variable(tf.zeros([1]))
    z = tf.matmul(x, w)  # + b
    hx = tf.nn.sigmoid(z)
    # hx = 1 / (1 + tf.exp(-z))

    # loss = (1 - y) * tf.log(1-hx) + y * tf.log(hx)
    # loss = keras.losses.binary_crossentropy(hx, y)
    loss = tf.losses.sigmoid_cross_entropy(hx, y)

    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    discounted = 0.99
    results = []

    for i in range(loop):
        state, _ = env.reset()
        done, stand = False, 0

        while not done:
            p = sess.run(hx, {x: state.reshape(1, -1)})
            action = int(p[0, 0] < 0.5)
            # action = int(p[0, 0] > 0.5)
            next_state, reward, done, _, _ = env.step(action)

            if done:
                p[0, 0] = -70
            else:
                p_next = sess.run(hx, {x: next_state.reshape(1, -1)})
                p[0, 0] = reward + discounted * p_next[0, 0]

            sess.run(train, {x: state.reshape(1, -1), y: p})
            state = next_state
            stand += reward

        print(i, stand)

        results.append(stand)
        if i % 10 == 0:
            print(i)

    # util.draw_q_table(q_table)

    print('count : ', i + 1, 'goal_times', results)
    plt.plot(range(len(results)), results)
    plt.tight_layout()
    plt.show()
    env.close()


# 퀴즈
# 시그모이드 케라스 버전을 텐서플로 버전으로 수정하세요

def q_network_dense(loop):
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1')

    x = tf.placeholder(tf.float32, shape=(None, 4))  # state
    y = tf.placeholder(tf.float32, shape=(None, 2))  # action

    w1 = tf.Variable(tf.random_uniform([4, 16]))
    b1 = tf.Variable(tf.zeros([16]))

    w2 = tf.Variable(tf.random_uniform([16, 2]))
    b2 = tf.Variable(tf.zeros([2]))
    # z = tf.matmul(x, w)  # + b
    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    hx = tf.matmul(r1, w2) + b2

    # loss = (1 - y) * tf.log(1-hx) + y * tf.log(hx)
    # loss = keras.losses.binary_crossentropy(hx, y)
    # loss = tf.losses.sigmoid_cross_entropy(hx, y)
    loss = tf.losses.mean_squared_error(hx, y)

    # train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    discounted = 0.99
    results = []

    for i in range(loop):
        state, _ = env.reset()
        done, stand = False, 0

        while not done:
            p = sess.run(hx, {x: state.reshape(1, -1)})
            action = int(p[0, 0] < 0.5)
            # action = int(p[0, 0] > 0.5)
            action = random_noise(i, env, actions=p[0])
            next_state, reward, done, _, _ = env.step(action)

            if done:
                p[0, action] = -70
            else:
                p_next = sess.run(hx, {x: next_state.reshape(1, -1)})
                p[0, action] = reward + discounted * np.max(p_next[0])

            sess.run(train, {x: state.reshape(1, -1), y: p})
            state = next_state
            stand += reward

        print(i, stand)

        results.append(stand)
        if i % 10 == 0:
            print(i)

    # util.draw_q_table(q_table)

    print('count : ', i + 1, 'goal_times', results)
    plt.plot(range(len(results)), results)
    plt.tight_layout()
    plt.show()
    env.close()


# q_network(20)
# q_network_sigmoid(2000)
q_network_dense(2000)
