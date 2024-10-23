# 2_4_q_learning.py

import gym
import matplotlib.pyplot as plt
import numpy as np
import util




# print(q_table)
# exit()
def simulation():
    # env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)
    env = gym.make('FrozenLake-v1', is_slippery=False)
    # env = gym.make('FrozenLake-v1', is_slippery=True)
    q_table = np.zeros((16, 4))
    for i in range(6):
        state, _ = env.reset()
        # print(state)
        for action in [2, 2, 1, 1, 1, 2]:
            next_state, reward, done, _, _ = env.step(action)
            # 요기에서 업데이트 하기
            # 퀴즈
            # q_table을 업데이트 하는 코드를 넣어주세요
            # q_table[state, action] = 현재보상 + 다음 상태에서의 누적 보상
            q_table[state, action] = reward + np.max(q_table[next_state])
            state = next_state
        # env.close()
        # print(q_table)

        util.draw_q_table(q_table)


# 현재 상태에서의 최대 보상에 대한 인덱스(LEFT, DOWN, RIGHT, UP) 반환
# 최대 값이 여러개 있을 때 랜덤하게 선택하기
def random_argmax(rewards):
    # return np.argmax(rewards) if sum(rewards) else np.random.randint(4)
    r_max = np.max(rewards)
    indices = np.nonzero(rewards == r_max)
    # print(indices, type(indices))
    # print(indices[0])

    return np.random.choice(indices[0])


# 퀴즈
# random_argmax 함수를 완성하고
# 2000번 반복 했을 때 횟루를 알려주세요
def q_learning():
    # env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)
    env = gym.make('FrozenLake-v1', is_slippery=False)
    # env = gym.make('FrozenLake-v1', is_slippery=True)
    q_table = np.zeros((16, 4))
    success = 0
    results = []
    for i in range(2000):
        state, _ = env.reset()

        done = False
        while not done:
            # print(state)
            action = random_argmax(q_table[state])
            next_state, reward, done, _, _ = env.step(action)
            # 요기에서 업데이트 하기
            # 퀴즈
            # q_table을 업데이트 하는 코드를 넣어주세요
            # q_table[state, action] = 현재보상 + 다음 상태에서의 누적 보상
            q_table[state, action] = reward + np.max(q_table[next_state])
            state = next_state

        success += reward
        results.append(reward)
        # if i % 10 == 0:
        #     print(i)
    util.draw_q_table(q_table)

        # 퀴즈
        # q_table로 부터 목표까지 이동하는 경로를 추출하세요
        # print(q_table)
    # done = False
    # while not done:
    #     action = np.argmax(q_table[state])
    #     state, _, done, _, _ = env.step(action)
    #     print(state)


    print([np.argmax(s) for s in q_table if sum(s)])    # 이동경로 구하기

    print('count : ', i + 1, 'goal_times', success)
    plt.plot(range(len(results)), results)
    # 퀴즈
    # hole, goal
    # 성공과 실패를 그래프로 그려주세요


    plt.show()



# simulation()
q_learning()
