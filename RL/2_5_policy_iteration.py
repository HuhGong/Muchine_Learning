# 3_7_policy_iteration.py
# 1. 교재에 있는 설명은 정책 테이블 사용 안함
#    사용할 경우 value iteration과 동일한 결과
# 2. 알고리즘의 정확성보다는 정책과 가치 테이블을 함께 사용하는 것에 초점
# 3. Q러닝에서 사용할 테이블 출력 코드 미리 작성(insert_text 함수만 없다)
#    util.py 파일에 있는 코드는 Q네트웍에서 생성한 가중치 Q테이블을 출력할 때까지 반복 사용 
import numpy as np
import util

np.set_printoptions(precision=2)

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def draw_table(policy_table, size):
    util.draw_value_table(util.make_memory_table(policy_table, size))
    print('\n')


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


# value_table을 이용해서 policy_table 업데이트
def policy_improvement(states, value_table, old_policy_table, size):
    reward = -1
    policy_table = np.zeros_like(old_policy_table)

    for s in states:
        values = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, size, action)
            values.append(reward + value_table[next_state])

        # 가장 큰 값들의 인덱스 찾기
        r_max = np.max(values)
        indices = np.nonzero(values == r_max)
        indices = indices[0]

        # 확률 계산
        policy_table[s, indices] = 1 / len(indices)

    return policy_table


# policy_table을 이용해서 value_table 업데이트
def policy_evaluation(states, old_value_table, policy_table, size):
    reward = -1
    value_table = np.zeros_like(old_value_table)

    for s in states:
        values = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, size, action)
            values.append(reward + old_value_table[next_state])

        # 교재에 나온 값 사용(항상 0.25). 154번째에서 수렴하지만, 정책 테이블을 업데이트하지 않는 코드
        # policy_table은 policy_improvement 함수에서 매번 업데이트
        # value_table[s] = np.sum([0.25 * v for a, v in enumerate(values)])

        # 값에 상관없이 액션에 따른 모든 보상을 반영
        prov = policy_table[s]
        value_table[s] = np.sum([prov[a] * v for a, v in enumerate(values)])

        # 업데이트가 정확한지 판단
        #     정책확률   미래보상  액션별 확률                                    업데이트 결과
        # print(prov, values, [prov[a] * v for a, v in enumerate(values)], value_table[s])

    return value_table


def policy_iteration(loop, size):
    states = [i for i in range(1, size * size - 1)]
    value_table = np.zeros(size * size)
    policy_table = np.zeros([size * size, 4])  # 4: all actions

    for i in range(loop):
        print(i + 1, '\n', value_table.reshape(-1, size))

        policy_table = policy_improvement(states, value_table, policy_table, size)
        value_table = policy_evaluation(states, value_table, policy_table, size)
        draw_table(policy_table, size)


policy_iteration(loop=4, size=4)
