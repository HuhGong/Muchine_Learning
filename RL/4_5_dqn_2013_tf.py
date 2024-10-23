# 4_5_dqn_2013_tf.py


import numpy as np
import random
import gym
from collections import deque
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def make_model(input_size, output_size):
    x = tf.placeholder(tf.float32, shape=(None, 4))  # state
    y = tf.placeholder(tf.float32, shape=(None, 2))  # action

    # (?, 16) = (?, 4) @ (4, 16)
    w1 = tf.Variable(tf.random_uniform([4, 16]))
    b1 = tf.Variable(tf.zeros([16]))

    w2 = tf.Variable(tf.random_uniform([16, 2]))
    b2 = tf.Variable(tf.zeros([2]))

    # model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mean_squared_error)
    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)
    z2 = tf.matmul(x, w2) + b2
    r2 = tf.nn.relu(z2)

    hx = tf.matmul(r1, w2) + b2
    # loss = (1 - y) * tf.log(1 - hx) + y * tf.log(hx)
    loss = tf.reduce_mean((hx - y) ** 2)
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    # loss = keras.losses.binary_crossentropy(hx, y)

    # train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train = tf.train.AdamOptimijer(0.01).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    return sess, train, hx, x, y


def annealing_epsilon(episode, min_e, max_e, target_episode):
    # episode가 50을 넘으면 0 반환. min_e : 0, max_e : 1
    slope = (min_e - max_e) / target_episode  # -0.02 = -1 / 50
    return max(min_e, slope * episode + max_e)  # max(0, -0.02 * epi + 1)


def dqn_2013_tf(sess, train, hx, x, y):
    env = gym.make('CartPole-v1')

    replay_buffer = deque(maxlen=50000)
    rewards_100 = deque(maxlen=100)

    max_episodes = 500
    for episode in range(max_episodes):
        state, _ = env.reset()

        e = annealing_epsilon(episode, 0.0, 1.0, max_episodes / 10)

        done, step_count, loss = False, 0, 0
        while not done:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                p = sess.run(hx, {x: np.reshape(state, [-1, 4])})
                action = np.argmax(p)

            next_state, reward, done, _, _ = env.step(action)

            if done:
                reward = -1

            replay_buffer.append((state, next_state, action, reward, done))

            state = next_state
            step_count += 1

            if len(replay_buffer) > 64:
                minibatch = random.sample(replay_buffer, 64)  # 2 28 20

                states = np.vstack([x[0] for x in minibatch])
                next_states = np.vstack([x[1] for x in minibatch])
                actions = np.array([x[2] for x in minibatch])
                rewards = np.array([x[3] for x in minibatch])
                dones = np.array([x[4] for x in minibatch])

                xx = states
                yy = sess.run(hx, {x: xx})

                next_rewards = sess.run(hx, {x: next_states}, verbose=0)

                yy[range(len(xx)), actions] = rewards + 0.99 * np.max(next_rewards, axis=1) * ~dones
                # yy[0, actions[0]] = rewards + 0.99 * np.max(next_rewards[0]) * ~dones # ~는 not, dones를 이용해서 if문을 대신 함
                # model.fit(xx, yy,epochs=1, batch_size=16, verbose=0)  # 64개니까 4번 함 16 * 4
                for n in range(0, 64, 16):
                    sess.run(train, {x: xx, y: yy})

        rewards_100.append(step_count)
        print('{} {} {}'.format(episode, step_count, int(np.mean(rewards_100))))

        if len(rewards_100) == rewards_100.maxlen:
            if np.mean(rewards_100) >= 195:
                break
    print(np.mean(rewards_100))
    env.close()


def bot_play_tf(sess, hx, x):
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()

    reward_sum, done = 0, False
    while not done:
        # env.render() # 버전이 오르니 없어도 작동
        p = sess.run(hx, {x: np.reshape(state, [-1, 4])})
        action = np.argmax(p)
        state, reward, done, _, _ = env.step(action)
        reward_sum += reward

    print('score :', reward_sum)
    env.close()


# 퀴즈
# dqn_2013 모델을 텐서플로 버전으로 수정하세요
#
sess, train, hx, x, y = make_model(4, 2)
dqn_2013_tf(sess, train, hx, x, y)
bot_play_tf(sess, hx, x)
