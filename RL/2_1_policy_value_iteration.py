# 2_1_policy_value_iteration.py
import numpy as np

np.set_printoptions(precision=2)  # 소수점 자리수 조절

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


# 퀴즈
# 다음 상태를 반환하는 함수를 만드세요
def get_next_state(state, action, size):
    row, col = state

    if action == LEFT and col > 0:
        col -= 1
    elif action == RIGHT and col < size - 1:
        col += 1
    elif action == UP and row > 0:
        row -= 1
    elif action == DOWN and row < size - 1:
        row += 1

    return row, col


def value_iteration(states, board, size):
    reward = -1
    grid = np.zeros_like(board)

    for s in states:
        values = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, action, size)
            values.append(reward + board[next_state])

        grid[s] = np.max(values)

    return grid


def policy_iteration(states, board, size):
    reward = -1
    grid = np.zeros_like(board)

    for s in states:
        values = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, action, size)
            values.append(reward + board[next_state])

        # grid[s] = np.sum([v * 0.25 for v in values]) # 평균
        # grid[s] = np.sum([v / 4 for v in values]) # 평균
        grid[s] = np.mean(values)  # 평균

    return grid


def show_iteration(iter_func, loop, size):
    board = np.zeros([size, size])
    # print(board)

    states = [(i, j) for i in range(size) for j in range(size)]
    states.pop(0)
    states.pop(-1)
    # print(states)

    for i in range(loop):
        # board = value_iteration(states, board, size)
        board = iter_func(states, board, size)
        # print(i + 1, '\n', board)
    show_direction(states, board, size)


def show_direction(states, board, size):
    policy = ['    ']

    for s in states:
        rewards = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, action, size)
            rewards.append(board[next_state])
        r_max = np.max(rewards)
        # 퀴즈
        # r_max를 사용해서 방향을 arrows에 추가하세요
        # ' DR ', 'L  R'
        arrows = ''
        arrows += 'L' if r_max == rewards[LEFT] else ' '
        arrows += 'D' if r_max == rewards[DOWN] else ' '
        arrows += 'R' if r_max == rewards[RIGHT] else ' '
        arrows += 'U' if r_max == rewards[UP] else ' '
        # arrows = ''.join('LDRU'[i] if r_max == rewards[direction] else ' '
        #                  for i, direction in enumerate([LEFT, DOWN, RIGHT, UP]))

        policy.append(arrows)

    policy.append('    ')  # teminal state
    policy = np.reshape(policy, (-1, 4))
    print(policy)


show_iteration(value_iteration, 3, size=4)
show_iteration(policy_iteration, 159, size=4)

# SGD : Stochastic Gradient Descent
# 60K개 데이터
# 1 : 6만개 모두 사용해서 1번 업데이트
# 2 : 100개 사용해서 1번 업데이터 (100개씩 600번)
