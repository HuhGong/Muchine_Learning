# 3_6_value_iteration.py
# value와 policy 코드가 각각 있는데, 난해하다. 전체적인 틀만 참고.
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/1-grid-world/2-value-iteration/value_iteration.py
import numpy as np

np.set_printoptions(precision=2)

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def show_policy(states, value_table, size):
    policy = ['    ']  # start
    for s in states:
        rewards = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, size, action)
            rewards.append(value_table[next_state])

        max_r = np.max(rewards)

        arrows = ''
        arrows += 'L' if max_r == rewards[LEFT] else ' '
        arrows += 'D' if max_r == rewards[DOWN] else ' '
        arrows += 'R' if max_r == rewards[RIGHT] else ' '
        arrows += 'U' if max_r == rewards[UP] else ' '

        policy.append(arrows)
    policy.append('    ')  # goal

    policy = np.reshape(policy, (-1, 4))
    print(policy)


def get_next_state(state, size, action):
    row, col = state // size, state % size

    if action == LEFT and col > 0:
        col -= 1
    elif action == RIGHT and col < size - 1:
        col += 1
    elif action == UP and row > 0:
        row -= 1
    elif action == DOWN and row < size - 1:
        row += 1

    return row * size + col


def value_evaluation(states, old_value_table, size):
    reward = -1
    value_table = np.zeros_like(old_value_table)

    for s in states:
        values = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, size, action)
            values.append(reward + old_value_table[next_state])

        # 최대값 선택
        value_table[s] = np.max(values)  # policy iteration과 다른 부분

    return value_table


def value_iteration(loop, size):
    states = [i for i in range(1, size * size - 1)]
    value_table = np.zeros(size * size)

    for i in range(loop):
        print(i + 1, '\n', value_table.reshape(-1, size))

        # policy iteration은 여기에 improvement 호출 들어간다 (policy 부분 추가)
        value_table = value_evaluation(states, value_table, size)

        show_policy(states, value_table, size)


value_iteration(loop=4, size=4)
