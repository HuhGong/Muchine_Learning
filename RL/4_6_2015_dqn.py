# 4_6_2015_dqn.py
import numpy as np
import random
import gym
from collections import deque
import keras


def make_model(input_size, output_size):
    model = keras.Sequential()
    model.add(keras.layers.Input([input_size]))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(output_size))

    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mean_squared_error)
    return model


def annealing_epsilon(episode, min_e, max_e, target_episode):
    # episode가 50을 넘으면 0 반환. min_e : 0, max_e : 1
    slope = (min_e - max_e) / target_episode  # -0.02 = -1 / 50
    return max(min_e, slope * episode + max_e)  # max(0, -0.02 * epi + 1)


def copy_weights(model_src, model_dst):
    w1 = model_src.get_layer(index=0).get_weights()
    w2 = model_src.get_layer(index=0).get_weights()

    model_dst.get_layer(index=0).set_weights(w1)
    model_dst.get_layer(index=0).set_weights(w2)


def dqn_2015(model, model_copy):
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
                action = np.argmax(model.predict(np.reshape(state, [1, -1]), verbose=0))

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
                yy = model.predict(xx, verbose=0)

                # ------------------------------------------------------- #
                next_rewards = model_copy.predict(next_states, verbose=0)
                # ------------------------------------------------------- #

                yy[range(len(xx)), actions] = rewards + 0.99 * np.max(next_rewards, axis=1) * ~dones
                # yy[0, actions[0]] = rewards + 0.99 * np.max(next_rewards[0]) * ~dones # ~는 not, dones를 이용해서 if문을 대신 함
                model.fit(xx, yy, batch_size=16, verbose=0)  # 64개니까 4번 함 16 * 4

                # ------------------------------------------------------- #
            if step_count % 10 == 9:
                copy_weights(model, model_copy)
                # ------------------------------------------------------- #
        rewards_100.append(step_count)
        print('{} {} {}'.format(episode, step_count, int(np.mean(rewards_100))))

        if len(rewards_100) == rewards_100.maxlen:
            if np.mean(rewards_100) >= 195:
                break
        # if episode % 100 == 99:
        #     model.save('models/dqn2015_{}.keras'.format(episode))
    print(np.mean(rewards_100))
    env.close()


def bot_play(model):
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()

    reward_sum, done = 0, False
    while not done:
        # env.render() # 버전이 오르니 없어도 작동
        action = np.argmax(model.predict(np.reshape(state, [1, 4]), verbose=0))
        state, reward, done, _, _ = env.step(action)
        reward_sum += reward

    print('score :', reward_sum)
    model.save('models/dpn2015.keras')
    env.close()


model = make_model(4, 2)
model_copy = make_model(4, 2)
copy_weights(model, model_copy)
dqn_2015(model, model_copy)

# model = keras.models.load_model(make_model('models/dqn2015.keras'))
# bot_play(model)